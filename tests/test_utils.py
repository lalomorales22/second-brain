"""Tests for utility functions."""

import math
import unittest
from semantic_gravity_memory.utils import (
    cosine_similarity,
    exponential_decay,
    reinforcement_boost,
    sigmoid,
    clamp,
    weighted_average,
    now_iso,
    parse_iso,
    seconds_between,
    is_expired,
    slugify,
    summarize_text,
    word_tokens,
    content_tokens,
    safe_json_dumps,
    safe_json_loads,
)


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=5)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=5)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0, places=5)

    def test_empty_vectors(self):
        self.assertEqual(cosine_similarity([], []), 0.0)

    def test_mismatched_lengths(self):
        self.assertEqual(cosine_similarity([1.0], [1.0, 2.0]), 0.0)

    def test_zero_vector(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 2.0]), 0.0)

    def test_known_value(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        # manual: dot=32, |a|=sqrt(14), |b|=sqrt(77)
        expected = 32.0 / (math.sqrt(14) * math.sqrt(77))
        self.assertAlmostEqual(cosine_similarity(a, b), expected, places=5)


class TestExponentialDecay(unittest.TestCase):
    def test_no_decay(self):
        self.assertAlmostEqual(exponential_decay(1.0, 0.0, 100.0), 1.0)

    def test_zero_time(self):
        self.assertAlmostEqual(exponential_decay(0.8, 0.5, 0.0), 0.8)

    def test_decays_over_time(self):
        val = exponential_decay(1.0, 0.1, 10.0)
        self.assertGreater(val, 0.0)
        self.assertLess(val, 1.0)
        self.assertAlmostEqual(val, math.exp(-1.0), places=5)

    def test_high_rate_fast_decay(self):
        val = exponential_decay(1.0, 1.0, 10.0)
        self.assertAlmostEqual(val, math.exp(-10.0), places=10)


class TestReinforcementBoost(unittest.TestCase):
    def test_boost_from_zero(self):
        result = reinforcement_boost(0.0, boost=0.15)
        self.assertAlmostEqual(result, 0.15)

    def test_diminishing_near_ceiling(self):
        result = reinforcement_boost(0.9, boost=0.15, ceiling=1.0)
        # should get 0.9 + 0.15 * 0.1 = 0.915
        self.assertAlmostEqual(result, 0.915, places=5)

    def test_at_ceiling(self):
        result = reinforcement_boost(1.0, boost=0.15)
        self.assertAlmostEqual(result, 1.0)


class TestSigmoid(unittest.TestCase):
    def test_zero(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=5)

    def test_large_positive(self):
        self.assertAlmostEqual(sigmoid(100.0), 1.0, places=5)

    def test_large_negative(self):
        self.assertAlmostEqual(sigmoid(-100.0), 0.0, places=5)

    def test_extreme_no_overflow(self):
        self.assertEqual(sigmoid(600.0), 1.0)
        self.assertEqual(sigmoid(-600.0), 0.0)


class TestClamp(unittest.TestCase):
    def test_within_range(self):
        self.assertEqual(clamp(0.5), 0.5)

    def test_below(self):
        self.assertEqual(clamp(-0.1), 0.0)

    def test_above(self):
        self.assertEqual(clamp(1.5), 1.0)

    def test_custom_range(self):
        self.assertEqual(clamp(5, lo=2, hi=8), 5)
        self.assertEqual(clamp(1, lo=2, hi=8), 2)
        self.assertEqual(clamp(10, lo=2, hi=8), 8)


class TestWeightedAverage(unittest.TestCase):
    def test_equal_weights(self):
        self.assertAlmostEqual(weighted_average([2.0, 4.0], [1.0, 1.0]), 3.0)

    def test_unequal_weights(self):
        self.assertAlmostEqual(weighted_average([10.0, 0.0], [3.0, 1.0]), 7.5)

    def test_empty(self):
        self.assertEqual(weighted_average([], []), 0.0)

    def test_zero_weights(self):
        self.assertEqual(weighted_average([1.0, 2.0], [0.0, 0.0]), 0.0)


class TestTimeHelpers(unittest.TestCase):
    def test_now_iso_format(self):
        ts = now_iso()
        # should be parseable
        parsed = parse_iso(ts)
        self.assertIsNotNone(parsed)

    def test_seconds_between(self):
        a = "2024-01-01T00:00:00"
        b = "2024-01-01T01:00:00"
        self.assertAlmostEqual(seconds_between(a, b), 3600.0)

    def test_seconds_between_reversed(self):
        a = "2024-01-01T01:00:00"
        b = "2024-01-01T00:00:00"
        self.assertAlmostEqual(seconds_between(a, b), 3600.0)

    def test_is_expired_past(self):
        self.assertTrue(is_expired("2020-01-01T00:00:00"))

    def test_is_expired_future(self):
        self.assertFalse(is_expired("2099-01-01T00:00:00"))

    def test_is_expired_none(self):
        self.assertFalse(is_expired(None))


class TestTextHelpers(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Hello World!"), "hello_world")

    def test_slugify_max_len(self):
        result = slugify("a" * 200, max_len=10)
        self.assertLessEqual(len(result), 10)

    def test_slugify_empty(self):
        self.assertEqual(slugify(""), "memory")

    def test_summarize_short(self):
        self.assertEqual(summarize_text("short text"), "short text")

    def test_summarize_long(self):
        long = "a " * 200
        result = summarize_text(long, max_len=20)
        self.assertLessEqual(len(result), 20)
        self.assertTrue(result.endswith("\u2026"))

    def test_word_tokens(self):
        tokens = word_tokens("Hello, world! This is a test-case.")
        self.assertIn("hello", tokens)
        self.assertIn("test-case", tokens)

    def test_content_tokens_removes_stopwords(self):
        tokens = content_tokens("this is a simple test with some words")
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)
        self.assertIn("simple", tokens)
        self.assertIn("test", tokens)


class TestJsonHelpers(unittest.TestCase):
    def test_dumps_dict(self):
        result = safe_json_dumps({"key": "value"})
        self.assertIn("key", result)

    def test_dumps_non_serializable(self):
        result = safe_json_dumps(object())
        self.assertEqual(result, "{}")

    def test_loads_valid(self):
        result = safe_json_loads('{"a": 1}')
        self.assertEqual(result, {"a": 1})

    def test_loads_invalid(self):
        result = safe_json_loads("not json", fallback=42)
        self.assertEqual(result, 42)

    def test_loads_none(self):
        result = safe_json_loads(None, fallback=[])
        self.assertEqual(result, [])

    def test_loads_empty_string(self):
        result = safe_json_loads("", fallback={})
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
