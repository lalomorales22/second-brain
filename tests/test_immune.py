"""Tests for the immune system — antibody creation, checking, suppression."""

import unittest

from semantic_gravity_memory.core.immune import (
    check_antibodies,
    create_antibody,
    deactivate_antibody,
)
from semantic_gravity_memory.storage.sqlite_backend import SQLiteBackend


class TestCreateAntibody(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")

    def tearDown(self):
        self.storage.close()

    def test_creates_and_returns_id(self):
        abid = create_antibody(
            self.storage,
            trigger_description="database migration",
            suppress_crystal_id=42,
            reason="gave wrong advice",
        )
        self.assertIsInstance(abid, int)
        self.assertGreater(abid, 0)

    def test_active_by_default(self):
        create_antibody(
            self.storage, "test trigger", suppress_crystal_id=1,
        )
        active = self.storage.active_antibodies()
        self.assertEqual(len(active), 1)
        self.assertTrue(active[0].active)

    def test_with_embedding(self):
        create_antibody(
            self.storage, "test", suppress_crystal_id=1,
            trigger_embedding=[0.5, 0.5, 0.0],
        )
        active = self.storage.active_antibodies()
        self.assertEqual(active[0].trigger_embedding, [0.5, 0.5, 0.0])


class TestCheckAntibodies(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")

    def tearDown(self):
        self.storage.close()

    def test_text_trigger_suppresses(self):
        create_antibody(
            self.storage,
            trigger_description="migration",
            suppress_crystal_id=10,
            reason="bad advice",
        )
        suppressions = check_antibodies(
            self.storage,
            query_text="Let's talk about database migration",
            query_embedding=None,
            candidate_crystal_ids={10, 20, 30},
        )
        self.assertEqual(len(suppressions), 1)
        self.assertEqual(suppressions[0]["crystal_id"], 10)
        self.assertEqual(suppressions[0]["reason"], "bad advice")

    def test_no_suppression_if_crystal_not_candidate(self):
        create_antibody(
            self.storage, "migration", suppress_crystal_id=10,
        )
        suppressions = check_antibodies(
            self.storage,
            query_text="migration plan",
            query_embedding=None,
            candidate_crystal_ids={20, 30},  # 10 not here
        )
        self.assertEqual(len(suppressions), 0)

    def test_no_suppression_if_trigger_no_match(self):
        create_antibody(
            self.storage, "migration", suppress_crystal_id=10,
        )
        suppressions = check_antibodies(
            self.storage,
            query_text="The weather is nice",
            query_embedding=None,
            candidate_crystal_ids={10},
        )
        self.assertEqual(len(suppressions), 0)

    def test_embedding_trigger_suppresses(self):
        create_antibody(
            self.storage,
            trigger_description="unrelated text",
            suppress_crystal_id=5,
            trigger_embedding=[1.0, 0.0, 0.0],
        )
        suppressions = check_antibodies(
            self.storage,
            query_text="totally different words",
            query_embedding=[0.98, 0.02, 0.0],  # very similar embedding
            candidate_crystal_ids={5},
            similarity_threshold=0.9,
        )
        self.assertEqual(len(suppressions), 1)

    def test_embedding_no_match(self):
        create_antibody(
            self.storage,
            trigger_description="nope",
            suppress_crystal_id=5,
            trigger_embedding=[1.0, 0.0, 0.0],
        )
        suppressions = check_antibodies(
            self.storage,
            query_text="nada",
            query_embedding=[0.0, 0.0, 1.0],  # orthogonal
            candidate_crystal_ids={5},
            similarity_threshold=0.9,
        )
        self.assertEqual(len(suppressions), 0)

    def test_multiple_antibodies(self):
        create_antibody(self.storage, "alpha", suppress_crystal_id=1)
        create_antibody(self.storage, "beta", suppress_crystal_id=2)
        suppressions = check_antibodies(
            self.storage,
            query_text="alpha beta gamma",
            query_embedding=None,
            candidate_crystal_ids={1, 2, 3},
        )
        self.assertEqual(len(suppressions), 2)
        suppressed_ids = {s["crystal_id"] for s in suppressions}
        self.assertEqual(suppressed_ids, {1, 2})


class TestDeactivateAntibody(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")

    def tearDown(self):
        self.storage.close()

    def test_deactivation_removes_from_active(self):
        abid = create_antibody(self.storage, "test", suppress_crystal_id=1)
        self.assertEqual(len(self.storage.active_antibodies()), 1)
        deactivate_antibody(self.storage, abid)
        self.assertEqual(len(self.storage.active_antibodies()), 0)


if __name__ == "__main__":
    unittest.main()
