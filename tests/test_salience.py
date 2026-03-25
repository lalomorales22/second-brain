"""Tests for salience scoring."""

import unittest

from semantic_gravity_memory.core.salience import score_salience
from semantic_gravity_memory.models import SalienceVector


class TestPracticalSalience(unittest.TestCase):
    def test_task_text_scores_high(self):
        sv = score_salience("I need to fix the bug and deploy by Friday")
        self.assertGreater(sv.practical, 0.2)

    def test_non_task_text_scores_low(self):
        sv = score_salience("The sunset was beautiful")
        self.assertLess(sv.practical, 0.2)

    def test_imperative_boost(self):
        sv = score_salience("please help me debug this error")
        self.assertGreater(sv.practical, 0.15)


class TestEmotionalSalience(unittest.TestCase):
    def test_emotional_text(self):
        sv = score_salience("I'm really stressed and worried about this deadline")
        self.assertGreater(sv.emotional, 0.2)

    def test_exclamation_boost(self):
        sv_calm = score_salience("This is great")
        sv_excited = score_salience("This is great!!!")
        self.assertGreater(sv_excited.emotional, sv_calm.emotional)

    def test_caps_boost(self):
        sv = score_salience("This is ABSOLUTELY CRITICAL")
        self.assertGreater(sv.emotional, 0.1)

    def test_neutral_text_low_emotional(self):
        sv = score_salience("The function returns a list of integers")
        self.assertLess(sv.emotional, 0.15)


class TestTemporalSalience(unittest.TestCase):
    def test_urgent_text(self):
        sv = score_salience("I need this done today, it's urgent")
        self.assertGreater(sv.temporal, 0.2)

    def test_deadline_with_day_name(self):
        sv = score_salience("The report is due by Friday")
        self.assertGreater(sv.temporal, 0.3)

    def test_no_temporal_markers(self):
        sv = score_salience("Python is a programming language")
        self.assertLess(sv.temporal, 0.15)


class TestUncertaintySalience(unittest.TestCase):
    def test_single_question(self):
        sv = score_salience("How do I set up the database?")
        self.assertGreater(sv.uncertainty, 0.1)

    def test_multiple_questions_higher(self):
        sv_one = score_salience("What is this?")
        sv_many = score_salience("What is this? How does it work? Why?")
        self.assertGreater(sv_many.uncertainty, sv_one.uncertainty)

    def test_uncertainty_words(self):
        sv = score_salience("I'm not sure, maybe we should try a different approach")
        self.assertGreater(sv.uncertainty, 0.2)


class TestIdentitySalience(unittest.TestCase):
    def test_non_general_state_boosts(self):
        sv = score_salience("some random text", self_state="builder")
        self.assertGreater(sv.identity, 0.1)

    def test_general_state_no_boost(self):
        sv = score_salience("some random text", self_state="general")
        self.assertLessEqual(sv.identity, 0.15)

    def test_identity_keywords(self):
        sv = score_salience("I'm a backend developer, my role is to build APIs")
        self.assertGreater(sv.identity, 0.1)


class TestNoveltySalience(unittest.TestCase):
    def test_novelty_keywords(self):
        sv = score_salience("I just discovered something new and surprising")
        self.assertGreater(sv.novelty, 0.1)

    def test_novelty_from_embeddings_orthogonal(self):
        recent = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]
        query = [0.0, 0.0, 1.0]
        sv = score_salience("test text", recent_embeddings=recent, query_embedding=query)
        self.assertGreater(sv.novelty, 0.3)

    def test_novelty_from_embeddings_similar(self):
        recent = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]
        query = [0.95, 0.05, 0.0]
        sv = score_salience("test text", recent_embeddings=recent, query_embedding=query)
        # Very similar to recent → low novelty from embeddings
        self.assertLess(sv.novelty, 0.3)

    def test_no_embeddings_no_crash(self):
        sv = score_salience("test", recent_embeddings=None, query_embedding=None)
        self.assertIsInstance(sv, SalienceVector)


class TestOutputType(unittest.TestCase):
    def test_returns_salience_vector(self):
        sv = score_salience("test")
        self.assertIsInstance(sv, SalienceVector)

    def test_all_dimensions_clamped(self):
        sv = score_salience(
            "I need to urgently fix this critical bug NOW!!! "
            "I hate it??? Not sure what to do! HELP!!!"
        )
        for dim in ("emotional", "practical", "temporal", "uncertainty", "identity", "novelty"):
            val = getattr(sv, dim)
            self.assertGreaterEqual(val, 0.0, f"{dim} below 0")
            self.assertLessEqual(val, 1.0, f"{dim} above 1")

    def test_combined_score_works(self):
        sv = score_salience("I need to deploy this urgent fix today!")
        combined = sv.combined()
        self.assertGreater(combined, 0.0)


if __name__ == "__main__":
    unittest.main()
