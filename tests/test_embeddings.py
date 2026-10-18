"""Ollama embedder: inputs too large for one pass are split and combined (no network)."""
import io
import json
import math
import unittest
import urllib.error
from unittest.mock import patch

from semantic_gravity_memory.embeddings import ollama
from semantic_gravity_memory.embeddings.ollama import InputTooLarge, OllamaEmbedder, _split

LIMIT = 300  # characters the fake server accepts in one input
TOO_LARGE = ('{"error":"input (2192 tokens) is too large to process. increase the physical '
             'batch size (current batch size: 2048)"}')


class FakeOllama:
    """Rejects long inputs like Ollama does (HTTP 413); otherwise returns a vector that
    counts the letters a and b, so combined vectors can be checked."""

    def __init__(self):
        self.inputs = []
        self.status = 413

    def __call__(self, request, timeout):
        payload = json.loads(request.data)
        inputs = payload['input'] if isinstance(payload['input'], list) else [payload['input']]
        self.inputs.append(payload['input'])
        if any(len(text) > LIMIT for text in inputs):
            raise urllib.error.HTTPError(request.full_url, self.status, 'error', {}, io.BytesIO(TOO_LARGE.encode()))
        vectors = [[text.count('a') + 1e-3, text.count('b') + 1e-3, 1.0] for text in inputs]
        return io.BytesIO(json.dumps({'embeddings': vectors}).encode())


class SplitTests(unittest.TestCase):
    def test_cuts_at_a_line_break_near_the_middle(self):
        text = 'first line\nsecond line\nthird line\nfourth line'
        left, right = _split(text)
        self.assertEqual(left + right, text)
        self.assertTrue(left.endswith('\n'))
        self.assertLess(abs(len(left) - len(right)), len(text) // 2)

    def test_falls_back_to_spaces_then_the_middle(self):
        self.assertEqual(''.join(_split('word ' * 20)), 'word ' * 20)
        self.assertEqual(_split('x' * 10), ('x' * 5, 'x' * 5))


class EmbedderTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOllama()
        self.patch = patch.object(ollama.urllib.request, 'urlopen', self.fake)
        self.patch.start()
        self.embedder = OllamaEmbedder('embeddinggemma-2')

    def tearDown(self):
        self.patch.stop()

    def test_short_text_is_one_request_with_the_task_prefix(self):
        self.embedder.embed_document('short note')
        self.assertEqual(self.fake.inputs, ['title: none | text: short note'])

    def test_long_text_is_split_until_it_fits_and_every_part_keeps_the_prefix(self):
        text = '\n'.join(f'line {i} ' + 'z' * 30 for i in range(40))
        vector = self.embedder.embed_document(text)
        accepted = [t for t in self.fake.inputs if len(t) <= LIMIT]
        self.assertGreater(len(accepted), 2)
        self.assertTrue(all(t.startswith('title: none | text: ') for t in self.fake.inputs))
        self.assertEqual(''.join(t[len('title: none | text: '):] for t in accepted), text)  # nothing lost
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in vector)), 1.0, places=6)

    def test_halves_are_weighted_by_length(self):
        text = 'a' * 300 + '\n' + 'b' * 100
        vector = OllamaEmbedder('all-minilm').embed(text)  # no prefix
        self.assertGreater(vector[0], vector[1])  # the longer 'a' part dominates

    def test_queries_and_batches_also_survive_long_input(self):
        self.embedder.embed_query('q' * 1000)
        self.assertTrue(all(t.startswith('task: search result | query: ') for t in self.fake.inputs))
        vectors = self.embedder.embed_batch(['short', 'a' * 900])
        self.assertEqual(len(vectors), 2)

    def test_other_errors_are_not_retried(self):
        calls = []

        def crashed(request, timeout):
            calls.append(request)
            raise urllib.error.HTTPError(request.full_url, 500, 'error', {}, io.BytesIO(b'{"error":"model crashed"}'))
        with patch.object(ollama.urllib.request, 'urlopen', crashed):
            with self.assertRaises(RuntimeError) as context:
                self.embedder.embed_document('x' * 1000)
        self.assertNotIsInstance(context.exception, InputTooLarge)
        self.assertIn('model crashed', str(context.exception))
        self.assertEqual(len(calls), 1)  # not split and retried

    def test_tiny_input_that_still_fails_is_reported(self):
        with patch.object(ollama, 'MIN_SPLIT_CHARS', 10_000):
            with self.assertRaises(InputTooLarge):
                self.embedder.embed_document('x' * 1000)


if __name__ == '__main__':
    unittest.main()
