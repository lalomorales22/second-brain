"""Tests for metamemory — quality tracking and domain calibration."""

import unittest

from semantic_gravity_memory.core.metamemory import MetaMemory
from semantic_gravity_memory.models import Activation, Crystal, SalienceVector
from semantic_gravity_memory.storage.sqlite_backend import SQLiteBackend


class TestRecordFeedback(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.mm = MetaMemory(self.storage)

    def tearDown(self):
        self.storage.close()

    def test_stores_feedback(self):
        self.mm.record_feedback(activation_id=1, quality=0.9, self_state="builder")
        log = self.mm.feedback_log()
        self.assertEqual(len(log), 1)
        self.assertAlmostEqual(log[0]["quality"], 0.9)
        self.assertEqual(log[0]["self_state"], "builder")

    def test_multiple_entries(self):
        self.mm.record_feedback(1, 0.8, "builder")
        self.mm.record_feedback(2, 0.6, "founder")
        self.mm.record_feedback(3, 0.9, "builder")
        log = self.mm.feedback_log()
        self.assertEqual(len(log), 3)


class TestDomainConfidence(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.mm = MetaMemory(self.storage)

    def tearDown(self):
        self.storage.close()

    def test_default_neutral(self):
        conf = self.mm.domain_confidence("unknown_state")
        self.assertAlmostEqual(conf, 0.5)

    def test_single_feedback(self):
        self.mm.record_feedback(1, 0.8, "builder")
        conf = self.mm.domain_confidence("builder")
        self.assertAlmostEqual(conf, 0.8)

    def test_average_across_feedbacks(self):
        self.mm.record_feedback(1, 1.0, "builder")
        self.mm.record_feedback(2, 0.6, "builder")
        conf = self.mm.domain_confidence("builder")
        self.assertAlmostEqual(conf, 0.8)

    def test_separate_domains(self):
        self.mm.record_feedback(1, 0.9, "builder")
        self.mm.record_feedback(2, 0.5, "founder")
        self.assertAlmostEqual(self.mm.domain_confidence("builder"), 0.9)
        self.assertAlmostEqual(self.mm.domain_confidence("founder"), 0.5)

    def test_all_domain_confidences(self):
        self.mm.record_feedback(1, 0.9, "builder")
        self.mm.record_feedback(2, 0.5, "founder")
        all_conf = self.mm.all_domain_confidences()
        self.assertIn("builder", all_conf)
        self.assertIn("founder", all_conf)
        self.assertAlmostEqual(all_conf["builder"], 0.9)


class TestRetrievalHistory(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.mm = MetaMemory(self.storage)

    def tearDown(self):
        self.storage.close()

    def test_crystal_recall_counts(self):
        # Insert some activations
        self.storage.insert_activation(Activation(
            query="q1", crystal_ids=[1, 2, 3],
        ))
        self.storage.insert_activation(Activation(
            query="q2", crystal_ids=[2, 3, 4],
        ))
        self.storage.insert_activation(Activation(
            query="q3", crystal_ids=[3],
        ))
        counts = self.mm.crystal_recall_counts()
        self.assertEqual(counts.get(1), 1)
        self.assertEqual(counts.get(2), 2)
        self.assertEqual(counts.get(3), 3)
        self.assertEqual(counts.get(4), 1)

    def test_never_recalled(self):
        self.storage.insert_crystal(Crystal(
            title="lonely", theme="t", summary="s",
        ))
        self.storage.insert_crystal(Crystal(
            title="popular", theme="t", summary="s",
        ))
        self.storage.insert_activation(Activation(
            query="q", crystal_ids=[2],
        ))
        never = self.mm.never_recalled_crystals()
        self.assertIn(1, never)
        self.assertNotIn(2, never)

    def test_most_recalled(self):
        self.storage.insert_activation(Activation(query="q", crystal_ids=[5, 5, 3]))
        self.storage.insert_activation(Activation(query="q", crystal_ids=[5]))
        top = self.mm.most_recalled_crystals(limit=2)
        self.assertEqual(top[0], 5)

    def test_empty_history(self):
        counts = self.mm.crystal_recall_counts()
        self.assertEqual(counts, {})


if __name__ == "__main__":
    unittest.main()
