"""Topic words and word bridges for the map."""
import unittest

from semantic_gravity_memory import keywords


def doc(*parts, filler=60):
    """Text where `parts` recur, padded with everyday words that should never surface."""
    return ' '.join(list(parts) * 6 + ['yeah you know going really the people'] * filler)


class TokenTests(unittest.TestCase):
    def test_skips_filler_and_marks_names(self):
        tokens = list(keywords.tokens('Yeah so the Pentagon briefed Congress about UAP. Pentagon again.'))
        words = [w for w, _ in tokens]
        self.assertNotIn('yeah', words)
        self.assertIn(('pentagon', True), tokens)  # capitalized mid-sentence
        self.assertIn(('uap', True), tokens)  # all capitals
        self.assertIn(('briefed', False), tokens)
        self.assertIn(('pentagon', False), tokens)  # sentence start is not evidence of a name


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.documents = {
            'source_a': doc('the Pentagon program Grusch testified', 'UAP craft recovered by Lockheed'),
            'source_b': doc('Grusch told Congress about UAP', 'recovered craft and Lockheed contracts'),
            'source_c': doc('LoRA adapters fine-tune a transformer', 'adapters merge into transformer weights'),
        }

    def test_each_item_gets_its_distinctive_words(self):
        top, _ = keywords.profiles(self.documents)
        self.assertIn('adapters', [w for w, _ in top['source_c']])
        self.assertNotIn('people', [w for w, _ in top['source_a']])
        self.assertEqual(top['source_a'][0][1], 1.0)  # weights are relative to the best word

    def test_bridges_name_the_shared_topic_and_skip_unrelated_pairs(self):
        _, stats = keywords.profiles(self.documents)
        links = keywords.bridges(stats)
        self.assertEqual([(b['a'], b['b']) for b in links], [('source_a', 'source_b')])
        self.assertTrue({'grusch', 'uap', 'lockheed'} & set(links[0]['words']))

    def test_skip_pairs_are_never_bridged(self):
        _, stats = keywords.profiles(self.documents)
        self.assertEqual(keywords.bridges(stats, skip={('source_b', 'source_a')}), [])

    def test_plurals_do_not_take_two_places(self):
        self.assertEqual(keywords.distinct([('ufo', 3), ('ufos', 2), ('craft', 1)], 3), [('ufo', 3), ('craft', 1)])


if __name__ == '__main__':
    unittest.main()
