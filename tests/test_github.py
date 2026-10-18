"""GitHub repository import: link parsing, file selection, passages, and download errors (no network)."""
import io
import tarfile
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from pathlib import Path

from semantic_gravity_memory import downloads, github
from semantic_gravity_memory.github import RepoRef, parse_repo, read_archive

COMMIT = 'c080742af4b8fac3e073ba210bd644dade1c1ca5'


def archive(files, commit=COMMIT, top='demo-HEAD', extra=()):
    """An in-memory .tar.gz shaped like GitHub's: a pax commit header and one top folder."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz', format=tarfile.PAX_FORMAT,
                      pax_headers={'comment': commit} if commit else {}) as tar:
        folder = tarfile.TarInfo(top)
        folder.type = tarfile.DIRTYPE
        tar.addfile(folder)
        for path, content in files.items():
            data = content.encode() if isinstance(content, str) else content
            info = tarfile.TarInfo(f'{top}/{path}')
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for info in extra:
            tar.addfile(info)
    return buffer.getvalue()


def code(lines, width=40):
    return '\n'.join(f'line_{i} = "{"x" * width}"' for i in range(1, lines + 1))


class ParseRepoTests(unittest.TestCase):
    def test_accepted_link_forms(self):
        cases = {
            'https://github.com/psf/requests': ('psf', 'requests', '', ''),
            'github.com/psf/requests.git': ('psf', 'requests', '', ''),
            'https://www.github.com/psf/requests/': ('psf', 'requests', '', ''),
            'git@github.com:psf/requests.git': ('psf', 'requests', '', ''),
            'https://github.com/psf/requests/tree/main/src/requests': ('psf', 'requests', 'main', 'src/requests'),
            'https://github.com/psf/requests/blob/v2.32.0/README.md': ('psf', 'requests', 'v2.32.0', 'README.md'),
            'https://github.com/a-b/c.d_e/tree/dev': ('a-b', 'c.d_e', 'dev', ''),
        }
        for link, expected in cases.items():
            ref = parse_repo(link)
            self.assertEqual((ref.owner, ref.repo, ref.ref, ref.path), expected, link)

    def test_rejected_links(self):
        for link in ('', 'https://gitlab.com/a/b', 'https://github.com/psf', 'https://github.com/psf/requests/issues/1',
                     'https://github.com/orgs/psf', 'https://evil-github.com/a/b', 'https://github.com/a/b/tree/main/../x',
                     'https://github.com/a/b/tree/..', 'not a link'):
            with self.assertRaises(ValueError, msg=link):
                parse_repo(link)


class SelectionTests(unittest.TestCase):
    FILES = {
        'README.md': '# Demo\nA demo project.',
        'src/app.py': 'def main():\n    return 42\n',
        'src/util.js': 'export const two = () => 2;\n',
        'tests/test_app.py': 'def test_main():\n    assert True\n',
        'docs/guide.md': 'Guide text.',
        'node_modules/left-pad/index.js': 'module.exports = 1;',
        'dist/bundle.js': 'var a=1;',
        'package-lock.json': '{}',
        'assets/logo.png': b'\x89PNG\r\n\x1a\n\x00\x00binary',
        'assets/icon.svg': '<svg><path d="M12.5 3.2L4.1 9.8 12.5 16.4z"/></svg>',
        'web/app.min.js': 'var a=1;',
        '.env': 'SECRET=1',
        '.github/workflows/ci.yml': 'on: push\n',
        'data/huge.csv': 'a,b\n' * 40000,
        'weird.txt': b'\xff\xfe not utf-8 \x00',
        'empty.py': '   \n',
        'generated.js': 'x' * 6000,
    }

    def test_keeps_docs_and_code_and_says_what_it_skipped(self):
        snapshot = read_archive(archive(self.FILES), RepoRef('me', 'demo'))
        paths = [f.path for f in snapshot.files]
        self.assertEqual(paths[0], 'README.md')
        self.assertEqual(sorted(paths), sorted(['README.md', 'src/app.py', 'src/util.js', 'tests/test_app.py',
                                                'docs/guide.md', '.github/workflows/ci.yml']))
        self.assertEqual(snapshot.commit, COMMIT)
        self.assertEqual(dict(snapshot.skipped), {
            'dependencies or build output': 2, 'lock files': 1, 'binary or media': 3,
            'minified or generated': 2, 'hidden files': 1, 'large data files': 1})

    def test_links_and_escaping_paths_are_never_followed(self):
        link = tarfile.TarInfo('demo-HEAD/linked.py')
        link.type = tarfile.SYMTYPE
        link.linkname = '/etc/passwd'
        escape = tarfile.TarInfo('demo-HEAD/../escape.py')
        escape.size = 0
        snapshot = read_archive(archive({'README.md': 'hi'}, extra=[link, escape]), RepoRef('me', 'demo'))
        self.assertEqual([f.path for f in snapshot.files], ['README.md'])

    def test_folder_link_imports_only_that_folder_and_its_readme_first(self):
        files = {'README.md': 'root', 'src/README.md': 'source readme', 'src/b.py': 'b = 1', 'src/a.py': 'a = 1',
                 'srcextra/c.py': 'c = 1', 'other/d.py': 'd = 1'}
        snapshot = read_archive(archive(files), RepoRef('me', 'demo', 'main', 'src'))
        self.assertEqual([f.path for f in snapshot.files], ['src/README.md', 'src/a.py', 'src/b.py'])

    def test_budget_keeps_readme_docs_and_code_before_tests(self):
        files = {'README.md': 'r' * 100, 'docs/a.md': 'd' * 100, 'src/main.py': 'c' * 100,
                 'tests/test_main.py': 't' * 100, 'config.yaml': 'y' * 100}
        snapshot = read_archive(archive(files), RepoRef('me', 'demo'), budget=320)
        self.assertEqual(sorted(f.path for f in snapshot.files), ['README.md', 'docs/a.md', 'src/main.py'])
        self.assertEqual(snapshot.skipped['over the size budget'], 2)

    def test_nothing_readable_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'No readable docs or code'):
            read_archive(archive({'logo.png': b'\x89PNG\x00'}), RepoRef('me', 'demo'))
        with self.assertRaisesRegex(ValueError, 'could not be read'):
            read_archive(b'not a tarball', RepoRef('me', 'demo'))


class PassageTests(unittest.TestCase):
    def test_every_piece_names_its_file_and_lines(self):
        snapshot = read_archive(archive({'README.md': '# Demo', 'big.py': code(150)}), RepoRef('me', 'demo'))
        passages = snapshot.passages()
        self.assertTrue(passages[0].startswith('# me/demo (GitHub repository)'))
        self.assertIn('commit c080742', passages[0])
        self.assertIn('\nbig.py', passages[0])  # the file tree
        ranges = [tuple(map(int, line.split('(lines ')[1].rstrip(')').split('–')))
                  for p in passages[1:] for line in p.splitlines() if line.startswith('File: big.py')]
        self.assertGreater(len(ranges), 2)
        self.assertEqual(ranges[0][0], 1)
        self.assertEqual(ranges[-1][1], 150)
        for (a1, b1), (a2, b2) in zip(ranges, ranges[1:]):
            self.assertEqual(a2, b1 + 1)  # contiguous, nothing lost
        self.assertTrue(all(len(p) <= github.PASSAGE_CHARS + 60 for p in passages))

    def test_small_files_share_passages(self):
        files = {f'pkg/m{i}.py': f'value_{i} = {i}' for i in range(20)}
        passages = read_archive(archive(files), RepoRef('me', 'demo')).passages()
        self.assertEqual(len(passages), 2)  # the overview and one passage holding all 20 files
        self.assertEqual(passages[1].count('File: pkg/'), 20)

    def test_document_fences_survive_backticks_in_code(self):
        snapshot = read_archive(archive({'README.md': 'Use ```python blocks``` here.'}), RepoRef('me', 'demo'))
        document = snapshot.document()
        self.assertIn('````md\nUse ```python blocks``` here.\n````', document)

    def test_names_metadata_and_dedupe_key(self):
        snapshot = read_archive(archive({'src/a.py': 'a = 1'}), RepoRef('me', 'demo', 'main', 'src'))
        self.assertEqual(snapshot.filename(), 'me · demo › src (GitHub).md')
        meta = snapshot.metadata()
        self.assertEqual((meta['origin'], meta['repo'], meta['commit'], meta['path']), ('github', 'me/demo', COMMIT, 'src'))
        self.assertEqual(meta['url'], f'https://github.com/me/demo/tree/{COMMIT}/src')
        self.assertEqual(meta['repo_key'], f'me/demo:src@{COMMIT}')


class FakeResponse(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data) if length is None else length)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.target = Path(self.folder.name) / 'repo.tar.gz'

    def tearDown(self):
        self.folder.cleanup()

    def test_streams_the_archive_to_disk(self):
        seen = []

        def fake(request, timeout):
            seen.append(request.full_url)
            return FakeResponse(b'tarball')
        with patch.object(github.urllib.request, 'urlopen', fake):
            self.assertEqual(github.download(RepoRef('me', 'demo'), self.target), 7)
            self.assertEqual(self.target.read_bytes(), b'tarball')
            github.download(RepoRef('me', 'demo', 'feature/x'), self.target)
        self.assertEqual(seen, ['https://codeload.github.com/me/demo/tar.gz/HEAD',
                                'https://codeload.github.com/me/demo/tar.gz/feature/x'])

    def test_missing_or_private_repository(self):
        error = urllib.error.HTTPError('u', 404, 'Not Found', {}, None)
        with patch.object(github.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(ValueError, 'no public repository me/demo.*Private'):
                github.download(RepoRef('me', 'demo'), self.target)

    def test_over_10_gb_stops_and_leaves_no_partial_file(self):
        with patch.object(github.urllib.request, 'urlopen',
                          return_value=FakeResponse(b'', length=github.MAX_DOWNLOAD + 1)):
            with self.assertRaisesRegex(ValueError, r'over the 10\.0 GB limit'):
                github.download(RepoRef('me', 'demo'), self.target)
        self.assertFalse(self.target.exists())

    def test_not_enough_disk_space(self):
        with patch.object(github.urllib.request, 'urlopen', return_value=FakeResponse(b'x' * 100)),                 patch.object(downloads, 'free_space', return_value=downloads.SPARE_DISK + 10):
            with self.assertRaisesRegex(ValueError, 'Not enough disk space'):
                github.download(RepoRef('me', 'demo'), self.target)
        self.assertFalse(self.target.exists())

    def test_fetch_reads_the_downloaded_archive(self):
        data = archive({'README.md': '# Demo', 'src/a.py': 'a = 1'})
        with patch.object(github.urllib.request, 'urlopen', return_value=FakeResponse(data)):
            snapshot = github.fetch_repository('https://github.com/me/demo')
        self.assertEqual([f.path for f in snapshot.files], ['README.md', 'src/a.py'])

    def test_sizes_read_naturally(self):
        self.assertEqual([downloads.human(n) for n in (512, 1536, 5 * 1024 ** 2, 10 * 1024 ** 3)],
                         ['512 bytes', '1.5 KB', '5.0 MB', '10.0 GB'])


if __name__ == '__main__':
    unittest.main()
