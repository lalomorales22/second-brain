"""Hugging Face dataset import: links, download, sampling rows and media (no network)."""
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from semantic_gravity_memory import huggingface
from semantic_gravity_memory.huggingface import DatasetRef, parse_dataset

SHA = 'aa13bc287fa6fcab6daf52f0dfb9994269ffea28'
JPEG = b'\xff\xd8\xff\xe0' + b'\x00' * 60
CARD = '---\nlicense: mit\npretty_name: Reviews\n---\n# Movie reviews\nShort film reviews labelled by sentiment.'


class FakeHub:
    """Answers the Hub, viewer, and file URLs the importer uses."""

    def __init__(self, files, viewer=True, tree_pages=1):
        self.files = files  # {path: bytes}
        self.viewer = viewer
        self.tree_pages = tree_pages
        self.seen = []

    def json(self, url, timeout=60):
        self.seen.append(url)
        if '/api/datasets/' in url and '/tree/' in url:
            entries = [{'type': 'file', 'path': p, 'size': len(d)} for p, d in self.files.items()]
            entries.append({'type': 'directory', 'path': 'data'})
            if self.tree_pages == 2 and 'cursor=' not in url:
                return entries[:1], '<https://huggingface.co/api/datasets/me/reviews/tree/x?cursor=2>; rel="next"'
            if self.tree_pages == 2:
                return entries[1:], ''
            return entries, ''
        if '/api/datasets/' in url:
            return {'id': 'me/reviews', 'sha': SHA, 'cardData': {'license': 'mit', 'pretty_name': 'Reviews',
                                                                 'task_categories': ['text-classification']}}, ''
        if not self.viewer:
            raise urllib.error.HTTPError(url, 404, 'Not Found', {}, io.BytesIO(b'{}'))
        if '/splits?' in url:
            return {'splits': [{'config': 'default', 'split': 'train'}]}, ''
        if '/rows?' in url:
            features = [{'name': 'text', 'type': {'_type': 'Value', 'dtype': 'string'}},
                        {'name': 'label', 'type': {'_type': 'ClassLabel', 'names': ['neg', 'pos']}},
                        {'name': 'poster', 'type': {'_type': 'Image'}}]
            rows = [{'row_idx': i, 'row': {'text': f'review number {i} about a quiet film',
                                           'label': i % 2, 'poster': {'src': f'https://cdn.example/{i}.jpg'}}}
                    for i in range(3)]
            return {'features': features, 'rows': rows, 'num_rows_total': 3}, ''
        raise AssertionError('unexpected URL ' + url)

    def stream(self, url, target, progress=None, label='', limit=None, headers=None, timeout=60,
               done_before=0, grand_total=None):
        self.seen.append(url)
        if url.startswith('https://cdn.example/'):
            data = JPEG
        else:
            path = url.split(f'/resolve/{SHA}/', 1)[1]
            data = self.files[urllib.parse.unquote(path)]
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(data)
        return len(data)


import urllib.parse  # noqa: E402  (used by FakeHub.stream)


class ParseTests(unittest.TestCase):
    def test_accepted_links(self):
        cases = {
            'https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes':
                ('cornell-movie-review-data/rotten_tomatoes', '', ''),
            'hf.co/datasets/me/reviews': ('me/reviews', '', ''),
            'https://huggingface.co/datasets/me/reviews/tree/main/data/train': ('me/reviews', 'main', 'data/train'),
            'https://huggingface.co/datasets/me/reviews/viewer/default/train': ('me/reviews', '', ''),
            'https://huggingface.co/datasets/imdb': ('imdb', '', ''),
        }
        for link, expected in cases.items():
            ref = parse_dataset(link)
            self.assertEqual((ref.repo_id, ref.revision, ref.path), expected, link)

    def test_rejected_links(self):
        for link, message in (('https://huggingface.co/openai/gpt-oss-20b', 'model or Space'),
                              ('https://huggingface.co/spaces/me/demo', 'model or Space'),
                              ('https://github.com/me/repo', 'Hugging Face dataset'),
                              ('https://huggingface.co/datasets/me/reviews/discussions/3', 'inside the dataset'),
                              ('https://huggingface.co/datasets/me/reviews/tree/main/../x', 'not supported'),
                              ('https://huggingface.co/datasets', 'Hugging Face dataset')):
            with self.assertRaisesRegex(ValueError, message, msg=link):
                parse_dataset(link)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name) / 'source'
        self.root.mkdir()

    def tearDown(self):
        self.folder.cleanup()

    def run_build(self, hub, ref=None):
        with patch.object(huggingface, 'get_json', hub.json), \
                patch.object(huggingface.downloads, 'stream_to_file', hub.stream):
            ref = ref or DatasetRef('me/reviews')
            return huggingface.build(ref, huggingface.fetch_info(ref), self.root)

    def test_downloads_everything_and_samples_readable_rows_and_images(self):
        hub = FakeHub({'README.md': CARD.encode(), 'data/train.parquet': b'PAR1' + b'\x00' * 100})
        snapshot = self.run_build(hub)
        self.assertEqual((self.root / 'files' / 'data' / 'train.parquet').read_bytes()[:4], b'PAR1')
        self.assertTrue(snapshot.card.startswith('# Movie reviews'))  # front matter removed
        self.assertEqual(snapshot.row_lines[0][1], 'Row 0 · text: review number 0 about a quiet film · label: neg')
        self.assertEqual(snapshot.columns, ['text (string)', 'label (class: neg, pos)', 'poster (image)'])
        images = [m for m in snapshot.media if m['media_kind'] == 'image']
        self.assertEqual(len(images), 3)
        self.assertTrue(all(Path(m['media_path']).read_bytes() == JPEG for m in images))
        self.assertIn('label: pos', images[1]['text'])
        passages = snapshot.passages()
        self.assertTrue(passages[0].startswith('# me/reviews (Hugging Face dataset)'))
        self.assertIn('License: mit', passages[0])
        self.assertIn('Splits: default/train (3 rows)', passages[0])
        self.assertTrue(any(p.startswith('Dataset card · me/reviews') for p in passages))
        meta = snapshot.metadata()
        self.assertEqual((meta['origin'], meta['sha'], meta['images'], meta['rows_sampled']), ('huggingface', SHA, 3, 3))
        self.assertEqual(meta['dataset_key'], f'me/reviews:@{SHA}')

    def test_without_the_viewer_rows_come_from_local_files(self):
        csv_bytes = b'text,label\nloved it,pos\nhated it,neg\n'
        hub = FakeHub({'README.md': b'# Card', 'train.csv': csv_bytes, 'pics/cat.jpg': JPEG}, viewer=False)
        snapshot = self.run_build(hub)
        self.assertEqual([line for _, line in snapshot.row_lines], ['Row 0 · text: loved it · label: pos',
                                                                    'Row 1 · text: hated it · label: neg'])
        self.assertEqual([Path(m['media_path']).name for m in snapshot.media], ['cat.jpg'])
        self.assertIn('Rows were sampled from the downloaded files.', snapshot.notes)

    def test_file_listing_follows_pages(self):
        hub = FakeHub({'README.md': b'# Card', 'a.csv': b'x,y\n1,2\n'}, viewer=False, tree_pages=2)
        snapshot = self.run_build(hub)
        self.assertEqual(sorted(p for p, _ in snapshot.files), ['README.md', 'a.csv'])

    def test_over_10_gb_is_refused_before_downloading(self):
        hub = FakeHub({'README.md': b'# Card'})
        with patch.object(huggingface, 'list_files', return_value=[('huge.parquet', 11 * 1024 ** 3)]), \
                patch.object(huggingface, 'get_json', hub.json), \
                patch.object(huggingface.downloads, 'stream_to_file', side_effect=AssertionError('downloaded')):
            with self.assertRaisesRegex(ValueError, r'11\.0 GB, over the 10 GB limit'):
                huggingface.build(DatasetRef('me/reviews'), {'sha': SHA}, self.root)

    def test_gated_dataset_explains_the_token(self):
        error = urllib.error.HTTPError('u', 401, 'Unauthorized', {}, io.BytesIO(b'{}'))
        with patch.object(huggingface.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(ValueError, 'gated or private.*HF_TOKEN'):
                huggingface.fetch_info(DatasetRef('me/secret'))

    def test_token_is_sent_when_configured(self):
        seen = []

        def fake(request, timeout):
            seen.append(request.headers.get('Authorization'))
            response = io.BytesIO(json.dumps({'sha': SHA}).encode())
            response.headers = {}
            return response
        with patch.dict(os.environ, {'HF_TOKEN': 'hf_test'}), \
                patch.object(huggingface.urllib.request, 'urlopen', fake):
            huggingface.fetch_info(DatasetRef('me/secret'))
        self.assertEqual(seen, ['Bearer hf_test'])


if __name__ == '__main__':
    unittest.main()
