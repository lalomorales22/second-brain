"""Regression coverage for source ingestion, model swaps, and conversation safety."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from semantic_gravity_memory import Memory
from semantic_gravity_memory.api.conversation import AnswerFilter, source_context
from semantic_gravity_memory.embeddings.indexing import install_embedder
from semantic_gravity_memory.embeddings.ollama import OllamaEmbedder
from semantic_gravity_memory.sources import SourceLibrary, chunk_text, MAX_BYTES
from semantic_gravity_memory.storage.sqlite_backend import SQLiteBackend


class FakeEmbedder:
    model = 'test'
    profile = 'test|v1'
    def embed_document(self, text):
        return [1., 0.] if 'PEAR-739' in text else [0., 1.]
    def embed_query(self, text):
        return [1., 0.]
    def embed_media(self, data, kind):
        return [0.5, 0.5]


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.storage = SQLiteBackend(str(Path(self.temp.name) / 'memory.db'))
        self.library = SourceLibrary(self.storage, Path(self.temp.name) / 'assets', FakeEmbedder())
    def tearDown(self):
        self.storage.close()
        self.temp.cleanup()

    def test_all_text_is_preserved_including_tail(self):
        text = 'ordinary background ' * 400 + '\nThe important code is PEAR-739.'
        result = self.library.ingest('notes.txt', text.encode())
        self.assertGreater(result['chunks'], 2)
        hits = self.library.search('What is the code?')
        self.assertIn('PEAR-739', hits[0]['text'])
        self.assertEqual(self.library.asset(source_id=result['id']).read_text(), text)

    def test_duplicate_content_does_not_make_duplicate_sources(self):
        first = self.library.ingest('one.txt', b'This is one note.')
        second = self.library.ingest('renamed.txt', b'This is one note.')
        self.assertEqual(first['id'], second['id'])
        self.assertTrue(second['duplicate'])
        self.assertEqual(len(self.library.list()), 1)

    def test_failed_embedding_leaves_no_half_import(self):
        with patch.object(self.library.embedder, 'embed_document', side_effect=RuntimeError('offline')):
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                self.library.ingest('broken.txt', b'Failure must not be saved.')
        self.assertEqual(self.library.list(), [])
        self.assertEqual(list(self.library.asset_dir.iterdir()), [])

    def test_commit_keeps_the_copy_saved_while_this_one_was_embedding(self):
        prepared = self.library.prepare('slow.txt', b'Same bytes, two imports.')
        first = self.library.ingest('fast.txt', b'Same bytes, two imports.')
        result = self.library.commit(prepared)
        self.assertEqual((result['id'], result['duplicate']), (first['id'], True))
        self.assertEqual(len(self.library.list()), 1)
        self.assertEqual([p.name for p in self.library.asset_dir.iterdir()], [first['id']])

    def test_search_within_one_source_returns_its_best_passages(self):
        first = self.library.ingest('first.txt', ('filler text ' * 300 + '\nThe code is PEAR-739.').encode())
        self.library.ingest('second.txt', b'The code is PEAR-739 here as well.')
        hits = self.library.search('What is the code?', limit=2, source_id=first['id'])
        self.assertTrue(hits)
        self.assertEqual({hit['source_id'] for hit in hits}, {first['id']})
        self.assertIn('PEAR-739', hits[0]['text'])

    def test_prebuilt_passages_replace_default_chunking(self):
        source = self.library.ingest('repo.md', b'# Whole document', passages=['File: a.py (lines 1-2)\nx', 'File: b.py (lines 1-1)\ny'])
        self.assertEqual([p['text'] for p in self.library.get(source['id'])['passages']],
                         ['File: a.py (lines 1-2)\nx', 'File: b.py (lines 1-1)\ny'])
        with self.assertRaisesRegex(ValueError, 'no text'):
            self.library.ingest('empty.md', b'# Other document', passages=['  '])

    def test_prepare_writes_nothing_to_the_database(self):
        prepared = self.library.prepare('later.txt', b'Not in the library until committed.')
        self.assertEqual(self.library.list(), [])
        self.assertEqual(self.library.commit(prepared)['name'], 'later.txt')

    def test_original_survives_reopening_database(self):
        source = self.library.ingest('keep.md', b'Keep the complete original.')
        self.storage.close()
        self.storage = SQLiteBackend(str(Path(self.temp.name) / 'memory.db'))
        library = SourceLibrary(self.storage, Path(self.temp.name) / 'assets', FakeEmbedder())
        self.assertEqual(library.get(source['id'])['name'], 'keep.md')
        self.assertEqual(library.asset(source_id=source['id']).read_bytes(), b'Keep the complete original.')

    def test_unrelated_embedding_profiles_are_not_compared(self):
        source = self.library.ingest('notes.txt', b'The secret is PEAR-739.')
        self.storage.conn.execute("UPDATE source_chunks SET profile='different-model'")
        self.storage.conn.commit()
        self.assertEqual(self.library.search('unrelated query'), [])

    def test_text_search_without_ollama(self):
        self.library.embedder = None
        self.library.ingest('notes.txt', b'Rainwater irrigation for apple trees.')
        self.assertEqual(len(self.library.search('irrigation')), 1)

    def test_media_requires_embeddings(self):
        self.library.embedder = None
        with self.assertRaisesRegex(ValueError, 'multimodal'):
            self.library.ingest('image.png', b'fake png')
        self.assertEqual(self.library.list(), [])

    def test_filename_cannot_escape_asset_directory(self):
        source = self.library.ingest('../../escape.txt', b'hello')
        self.assertEqual(source['name'], 'escape.txt')
        self.assertIn(self.library.asset_dir, self.library.asset(source_id=source['id']).parents)

    def test_unsupported_or_empty_input_rejected(self):
        for name, data in [('a.txt', b''), ('a.exe', b'bad')]:
            with self.assertRaises(ValueError):
                self.library.ingest(name, data)

    def test_chunking_has_overlap_and_covers_tail(self):
        text = ''.join(str(i % 10) for i in range(10000))
        chunks = chunk_text(text)
        self.assertTrue(chunks[-1].endswith(text[-300:]))
        for a, b in zip(chunks, chunks[1:]):
            self.assertEqual(a[-240:], b[:240])

    def test_image_gets_real_media_embedding_and_preview(self):
        with patch.object(self.library.embedder, 'embed_media', return_value=[.5,.5]) as embed:
            source = self.library.ingest('photo.png', b'png bytes')
        embed.assert_called_once_with(b'png bytes', 'image')
        self.assertEqual(source['passages'][0]['media_kind'], 'image')
        self.assertIn('/api/chunks/', source['passages'][0]['preview_url'])


class IndexingTests(unittest.TestCase):
    def setUp(self):
        self.memory = Memory(db_path=':memory:')
        self.memory.ingest('I prefer green notebooks.')
    def tearDown(self):
        self.memory.close()

    def test_failed_rebuild_preserves_old_vectors_and_model(self):
        with patch.object(OllamaEmbedder, 'embed_query', return_value=[1., 0.]), patch.object(OllamaEmbedder, 'embed_document', return_value=[1.,0.]):
            install_embedder(self.memory, 'old', 'http://localhost:11434/api')
        before = self.memory._storage.get_crystal(1).embedding
        with patch.object(OllamaEmbedder, 'embed_query', return_value=[0.,1.]), patch.object(OllamaEmbedder, 'embed_document', side_effect=RuntimeError('unavailable')):
            with self.assertRaises(RuntimeError):
                install_embedder(self.memory, 'new', 'http://localhost:11434/api')
        self.assertEqual(self.memory._engine.embedder.model, 'old')
        self.assertEqual(self.memory._storage.get_crystal(1).embedding, before)
        self.assertEqual(self.memory._storage.get_meta('embedding_model'), 'old')

    def test_successful_rebuild_updates_all_memory_vectors(self):
        with patch.object(OllamaEmbedder, 'embed_query', return_value=[0.,1.]), patch.object(OllamaEmbedder, 'embed_document', return_value=[0.,1.]):
            install_embedder(self.memory, 'new', 'http://localhost:11434/api')
        self.assertEqual(self.memory._storage.get_crystal(1).embedding, [0.,1.])
        self.assertEqual(self.memory._storage.get_event(1).embedding, [0.,1.])

    def test_disabling_model_persists_without_losing_vectors(self):
        with patch.object(OllamaEmbedder, 'embed_query', return_value=[1., 0.]), patch.object(OllamaEmbedder, 'embed_document', return_value=[1.,0.]):
            install_embedder(self.memory, 'old', 'http://localhost:11434/api')
        install_embedder(self.memory, '', 'http://localhost:11434/api')
        self.assertIsNone(self.memory._engine.embedder)
        self.assertEqual(self.memory._storage.get_meta('embedding_model'), '')
        self.assertEqual(self.memory._storage.get_meta('embedding_configured'), 'yes')
        self.assertEqual(self.memory._storage.get_crystal(1).embedding, [1.,0.])

    def test_enabling_same_model_indexes_memories_added_while_disabled(self):
        with patch.object(OllamaEmbedder, 'embed_query', return_value=[1., 0.]), patch.object(OllamaEmbedder, 'embed_document', return_value=[1.,0.]):
            install_embedder(self.memory, 'old', 'http://localhost:11434/api')
            install_embedder(self.memory, '', 'http://localhost:11434/api')
            self.memory.ingest('This was saved offline.')
            install_embedder(self.memory, 'old', 'http://localhost:11434/api')
        self.assertEqual(self.memory._storage.get_crystal(2).embedding, [1.,0.])


class StreamTests(unittest.TestCase):
    def test_reasoning_is_excluded_at_every_possible_split(self):
        text = '<think>private reasoning</think>The answer is 42.'
        for split in range(len(text) + 1):
            parser = AnswerFilter()
            output = parser.feed(text[:split]) + parser.feed(text[split:]) + parser.feed('', final=True)
            self.assertEqual(output, 'The answer is 42.', split)

    def test_character_by_character_stream(self):
        parser = AnswerFilter()
        output = ''.join(parser.feed(c) for c in 'Hello<think>hide this</think> world') + parser.feed('', final=True)
        self.assertEqual(output, 'Hello world')

    def test_unclosed_reasoning_is_not_an_answer(self):
        parser = AnswerFilter()
        self.assertEqual(parser.feed('<think>unfinished') + parser.feed('', final=True), '')

    def test_sources_are_labeled_and_timestamps_retained(self):
        prompt = source_context([{'name':'clip.mp4','start_seconds':12.5,'text':'Video frame.'}])
        self.assertIn('[S1] clip.mp4 at 12.5s', prompt)
        self.assertIn('not a transcript', prompt)


class APITests(unittest.TestCase):
    def setUp(self):
        import threading
        from semantic_gravity_memory.api import server
        self.server_module = server
        self.temp = tempfile.TemporaryDirectory()
        self.memory = Memory(db_path=str(Path(self.temp.name) / 'test.db'))
        self.library = SourceLibrary(self.memory._storage, Path(self.temp.name) / 'assets')
        self.original = server._state.copy()
        server._state.update(memory=self.memory, library=self.library, chat_model='chat-test',
                             embed_model='', last_scene={}, jobs={}, active_job=None, model_info={
                                 'chat-test': {'capabilities': ['completion']}})
        self.server = server.ThreadedHTTPServer(('127.0.0.1', 0), server.BrainHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.server_module._state.clear()
        self.server_module._state.update(self.original)
        self.memory.close()
        self.temp.cleanup()

    def call(self, path, data=None, headers=None):
        import urllib.request
        request = urllib.request.Request(self.base + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers=headers or {'Content-Type': 'application/json'})
        return urllib.request.urlopen(request, timeout=10)

    def test_chat_saves_user_but_never_promotes_assistant_to_fact(self):
        chunks = [{'message': {'content': '<thi'}}, {'message': {'content': 'nk>hidden</think>Answer.'}}, {'done': True}]
        with patch.object(self.server_module, 'chat_stream', return_value=iter(chunks)):
            with self.call('/api/answer', {'query': 'I prefer green notebooks.'}) as response:
                data = response.read().decode()
        self.assertIn('Answer.', data)
        self.assertNotIn('hidden', data)
        events = self.memory._storage.recent_events()
        self.assertEqual({event.actor for event in events}, {'user', 'assistant'})
        crystals = self.memory._storage.all_crystals()
        self.assertEqual(len(crystals), 1)
        self.assertIn('green notebooks', crystals[0].summary)
        with self.call('/api/history') as response:
            history = json.load(response)['messages']
        self.assertEqual([m['role'] for m in history], ['user', 'assistant'])

    def test_generation_failure_keeps_user_memory(self):
        with patch.object(self.server_module, 'chat_stream', side_effect=RuntimeError('offline')):
            with self.call('/api/answer', {'query': 'Remember my preference for green.'}) as response:
                data = response.read().decode()
        self.assertIn('offline', data)
        self.assertEqual(len(self.memory._storage.all_crystals()), 1)
        self.assertEqual(len(self.memory._storage.recent_events()), 1)

    def test_malformed_json_returns_error(self):
        import urllib.request
        import urllib.error
        request = urllib.request.Request(self.base + '/api/ingest', data=b'{bad', headers={'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(request)
        self.assertEqual(context.exception.code, 400)
        context.exception.close()

    def test_static_traversal_is_not_served(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.call('/static/%2e%2e/api/server.py')
        self.assertEqual(context.exception.code, 404)
        context.exception.close()

    def test_media_range_support_for_seeking(self):
        source = self.library.ingest('small.txt', b'0123456789')
        with self.call(source['url'], headers={'Range': 'bytes=3-5'}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b'345')
            self.assertTrue(response.headers['Content-Disposition'].startswith('attachment'))

    def test_import_job_and_duplicate(self):
        import urllib.request
        import time
        for attempt in range(2):
            request = urllib.request.Request(self.base + '/api/sources?name=test.txt', data=b'A stored original file.', headers={'Content-Type': 'application/octet-stream'})
            with urllib.request.urlopen(request) as response:
                job = json.load(response)
                self.assertEqual(response.status, 202)
            for _ in range(100):
                with self.call('/api/jobs/' + job['id']) as response:
                    job = json.load(response)
                if job['status'] != 'running':
                    break
                time.sleep(.01)
            self.assertEqual(job['status'], 'done', job)
        self.assertEqual(len(self.library.list()), 1)
        self.assertEqual(len(self.memory._storage.all_crystals()), 1)

    def wait_job(self, job):
        import time
        for _ in range(200):
            if job['status'] != 'running':
                break
            time.sleep(.01)
            with self.call('/api/jobs/' + job['id']) as response:
                job = json.load(response)
        return job

    def test_youtube_link_imports_transcript_once(self):
        from semantic_gravity_memory import youtube
        transcript = youtube.Transcript(video_id='dQw4w9WgXcQ', title='Podcast #7: Memory', channel='The Show',
                                        duration=120, language='en', language_name='English', captions='auto-generated',
                                        cues=[(0, 'Welcome to the show.'), (5, 'Today the code word is PEAR-739.')])
        with self.call('/api/config') as response:
            self.assertIn('youtube', json.load(response)['capabilities'])
        with patch.object(youtube, 'fetch_transcript', return_value=transcript) as fetch:
            with self.call('/api/sources/youtube', {'url': 'https://youtu.be/dQw4w9WgXcQ'}) as response:
                self.assertEqual(response.status, 202)
                job = self.wait_job(json.load(response))
            self.assertEqual(job['status'], 'done', job)
            source = job['result']
            self.assertEqual(source['name'], 'Podcast #7- Memory (YouTube).md')
            self.assertEqual((source['kind'], source['metadata']['origin'], source['metadata']['video_id']),
                             ('text', 'youtube', 'dQw4w9WgXcQ'))
            self.assertTrue(source['crystal_id'])
            self.assertIn('[00:00:00] Welcome to the show. Today the code word is PEAR-739.',
                          self.library.get(source['id'])['passages'][0]['text'])
            # The same video by another link form is recognised without fetching again.
            with self.call('/api/sources/youtube', {'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s'}) as response:
                again = self.wait_job(json.load(response))
            self.assertEqual(fetch.call_count, 1)
        self.assertEqual((again['status'], again['result']['id'], again['result']['duplicate']), ('done', source['id'], True))
        self.assertEqual(len(self.library.list()), 1)
        self.assertEqual(len(self.memory._storage.all_crystals()), 1)

    def test_search_and_library_answer_while_an_import_is_embedding(self):
        import threading
        import time
        import urllib.request
        started, release = threading.Event(), threading.Event()

        class SlowEmbedder(FakeEmbedder):
            def embed_document(self, text):
                started.set()
                release.wait(30)
                return super().embed_document(text)

        self.library.embedder = SlowEmbedder()
        try:
            request = urllib.request.Request(self.base + '/api/sources?name=episode.txt', data=b'A long transcript. ' * 50,
                                             headers={'Content-Type': 'application/octet-stream'})
            with urllib.request.urlopen(request) as response:
                job = json.load(response)
            self.assertTrue(started.wait(5), 'the import never started embedding')
            began = time.monotonic()
            with self.call('/api/search', {'query': 'anything'}) as response:
                self.assertEqual(response.status, 200)
            with self.call('/api/sources') as response:
                self.assertEqual(json.load(response)['sources'], [])  # nothing half-written is visible
            self.assertLess(time.monotonic() - began, 3, 'search or the library waited for the import')
        finally:
            release.set()
        job = self.wait_job(job)
        self.assertEqual(job['status'], 'done', job)
        self.assertTrue(job['result']['crystal_id'])
        self.assertEqual(len(self.library.list()), 1)

    def test_github_repository_imports_once_per_commit(self):
        from semantic_gravity_memory import github
        from tests.test_github import archive, code
        tarball = archive({'README.md': '# Demo\nThe secret word is PEAR-739.', 'src/app.py': code(120)})
        with patch.object(github, 'download', return_value=tarball) as download:
            with self.call('/api/sources/github', {'url': 'https://github.com/me/demo'}) as response:
                self.assertEqual(response.status, 202)
                job = self.wait_job(json.load(response))
            self.assertEqual(job['status'], 'done', job)
            source = job['result']
            self.assertEqual(source['name'], 'me · demo (GitHub).md')
            self.assertEqual((source['metadata']['origin'], source['metadata']['repo']), ('github', 'me/demo'))
            passages = self.library.get(source['id'])['passages']
            self.assertTrue(passages[0]['text'].startswith('# me/demo (GitHub repository)'))
            self.assertTrue(any('File: src/app.py (lines 1–' in p['text'] for p in passages))
            self.assertTrue(source['crystal_id'])
            with self.call('/api/sources/github', {'url': 'github.com/me/demo.git'}) as response:
                again = self.wait_job(json.load(response))
            self.assertEqual(download.call_count, 2)
        self.assertEqual((again['status'], again['result']['id'], again['result']['duplicate']), ('done', source['id'], True))
        self.assertEqual(len(self.library.list()), 1)

    def test_github_rejects_other_links_before_starting_a_job(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.call('/api/sources/github', {'url': 'https://gitlab.com/me/demo'})
        self.assertEqual(context.exception.code, 400)
        context.exception.close()
        self.assertIsNone(self.server_module._state['active_job'])

    def test_chat_about_a_pinned_source_uses_its_passages_and_connections(self):
        pinned = self.library.ingest('notes.txt', ('Unrelated opening line.\n' * 200 + 'The launch date is March 9.').encode())
        self.library.ingest('other.txt', b'A different document about gardening.')
        captured = []

        def fake_stream(url, model, messages, thinking):
            captured.append(messages)
            return iter([{'message': {'content': 'Answer.'}}, {'done': True}])
        focus = {'type': 'source', 'id': pinned['id'], 'title': 'notes.txt',
                 'connections': [{'title': 'Launch plan', 'relation': 'source_memory', 'type': 'crystal'}]}
        with patch.object(self.server_module, 'chat_stream', fake_stream):
            with self.call('/api/answer', {'query': 'When is it?', 'focus': focus}) as response:
                data = response.read().decode()
        self.assertIn('Answer.', data)
        system = captured[0][0]['content']
        self.assertIn('The user is asking about this item from their memory map', system)
        self.assertIn('Source: notes.txt', system)
        self.assertIn('- source memory → Launch plan (memory)', system)
        self.assertIn('[S1] notes.txt', system)  # its own passages lead the evidence
        self.assertIn('When is it?', [m['content'] for m in captured[0] if m['role'] == 'user'][-1])
        stored = [e.content for e in self.memory._storage.recent_events() if e.actor == 'user']
        self.assertEqual(stored, ['When is it?'])  # the message is saved exactly as typed

    def test_chat_ignores_malformed_focus(self):
        captured = []

        def fake_stream(url, model, messages, thinking):
            captured.append(messages)
            return iter([{'message': {'content': 'Fine.'}}, {'done': True}])
        with patch.object(self.server_module, 'chat_stream', fake_stream):
            for focus in ('text', {'type': 'admin', 'title': 'x'}, {'type': 'source', 'title': ''}):
                with self.call('/api/answer', {'query': 'Hello?', 'focus': focus}) as response:
                    response.read()
        self.assertTrue(all('memory map ===' not in m[0]['content'] for m in captured))

    def test_youtube_rejects_non_video_link_before_starting_a_job(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.call('/api/sources/youtube', {'url': 'https://www.youtube.com/playlist?list=PL123'})
        self.assertEqual(context.exception.code, 400)
        self.assertIn('single YouTube video', json.load(context.exception)['error'])
        context.exception.close()
        self.assertIsNone(self.server_module._state['active_job'])

    def test_failed_youtube_fetch_reports_error_and_saves_nothing(self):
        from semantic_gravity_memory import youtube
        with patch.object(youtube, 'fetch_transcript', side_effect=ValueError('YouTube: Video unavailable')):
            with self.call('/api/sources/youtube', {'url': 'https://youtu.be/dQw4w9WgXcQ'}) as response:
                job = self.wait_job(json.load(response))
        self.assertEqual((job['status'], job['error']), ('error', 'YouTube: Video unavailable'))
        self.assertEqual(self.library.list(), [])

    def test_cross_origin_write_rejected(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.call('/api/ingest', {'text': 'should not save'}, {'Origin': 'https://unrelated.example', 'Content-Type': 'application/json'})
        self.assertEqual(context.exception.code, 403)
        context.exception.close()
        self.assertEqual(self.memory._storage.all_crystals(), [])


class DocumentTests(unittest.TestCase):
    def test_docx_preserves_paragraphs(self):
        import zipfile
        from semantic_gravity_memory.sources import extract_document
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'note.docx'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>First paragraph.</w:t></w:r></w:p><w:p><w:r><w:t>Second paragraph.</w:t></w:r></w:p></w:body></w:document>')
            self.assertEqual(extract_document(path, '.docx'), 'First paragraph.\nSecond paragraph.')

    def test_pdf_with_text_and_scanned_pdf_failure(self):
        try:
            from pypdf import PdfWriter
            from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        except ImportError:
            self.skipTest('Optional pypdf extra is not installed.')
        from semantic_gravity_memory.sources import extract_document
        with tempfile.TemporaryDirectory() as folder:
            blank = Path(folder) / 'blank.pdf'
            writer = PdfWriter()
            page = writer.add_blank_page(width=300, height=300)
            writer.write(blank)
            with self.assertRaisesRegex(ValueError, 'selectable text'):
                extract_document(blank, '.pdf')
            font = DictionaryObject({NameObject('/Type'):NameObject('/Font'), NameObject('/Subtype'):NameObject('/Type1'), NameObject('/BaseFont'):NameObject('/Helvetica')})
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):font})})
            stream = DecodedStreamObject()
            stream.set_data(b'BT /F1 12 Tf 20 250 Td (The project deadline is November 14.) Tj ET')
            page[NameObject('/Contents')] = stream
            path = Path(folder) / 'text.pdf'
            writer.write(path)
            text = extract_document(path, '.pdf')
            self.assertIn('[Page 1]', text)
            self.assertIn('November 14', text)
