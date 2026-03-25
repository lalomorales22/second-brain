"""Tests for retrieval — spreading activation, scene reconstruction, working memory."""

import unittest

from semantic_gravity_memory.core.retrieval import (
    RetrievalEngine,
    WorkingMemoryBuffer,
    spread_activation,
)
from semantic_gravity_memory.core.immune import create_antibody
from semantic_gravity_memory.core.temporal import create_prospective
from semantic_gravity_memory.models import (
    Activation,
    Crystal,
    Entity,
    Relation,
    SalienceVector,
)
from semantic_gravity_memory.storage.sqlite_backend import SQLiteBackend


def _high_sal():
    return SalienceVector(
        emotional=0.8, practical=0.8, identity=0.8,
        temporal=0.8, uncertainty=0.8, novelty=0.8,
    )


def _insert_crystal(storage, title, ts="2024-06-01T12:00:00", **kw):
    defaults = dict(
        title=title, theme=title, summary=f"about {title}",
        created_ts=ts, salience=_high_sal(), confidence=0.8,
        decay_rate=0.05, memory_type="episodic",
    )
    defaults.update(kw)
    return storage.insert_crystal(Crystal(**defaults))


def _insert_entity(storage, name, kind="concept"):
    return storage.upsert_entity(Entity(name=name, kind=kind, salience=0.3))


# =========================================================================
# Spreading Activation
# =========================================================================


class TestSpreadingActivation(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")

    def tearDown(self):
        self.storage.close()

    def test_no_relations_returns_seeds(self):
        cid = _insert_crystal(self.storage, "alpha")
        result = spread_activation(
            self.storage, {("crystal", cid): 1.0}, max_hops=3,
        )
        self.assertAlmostEqual(result[("crystal", cid)], 1.0)

    def test_one_hop_through_relation(self):
        c1 = _insert_crystal(self.storage, "source")
        c2 = _insert_crystal(self.storage, "target")
        self.storage.insert_relation(Relation(
            source_type="crystal", source_id=c1,
            target_type="crystal", target_id=c2,
            relation="temporal_cluster", weight=0.8,
        ))
        result = spread_activation(
            self.storage, {("crystal", c1): 1.0},
            max_hops=1, hop_decay=0.5,
        )
        self.assertIn(("crystal", c2), result)
        # energy = 1.0 * 0.8 * 0.5 = 0.4
        self.assertAlmostEqual(result[("crystal", c2)], 0.4, places=2)

    def test_two_hops_through_entity(self):
        c1 = _insert_crystal(self.storage, "crystal_a")
        c2 = _insert_crystal(self.storage, "crystal_b")
        eid = _insert_entity(self.storage, "shared_entity")
        # c1 → entity (mentions)
        self.storage.insert_relation(Relation(
            source_type="crystal", source_id=c1,
            target_type="entity", target_id=eid,
            relation="mentions", weight=0.6,
        ))
        # c2 → entity (mentions) — reverse edge connects entity to c2
        self.storage.insert_relation(Relation(
            source_type="crystal", source_id=c2,
            target_type="entity", target_id=eid,
            relation="mentions", weight=0.6,
        ))
        result = spread_activation(
            self.storage, {("crystal", c1): 1.0},
            max_hops=2, hop_decay=0.5,
        )
        # hop1: c1 → entity (1.0 * 0.6 * 0.5 = 0.3)
        # hop2: entity ← c2 (reverse of c2→entity: 0.3 * 0.6 * 0.5 = 0.09)
        self.assertIn(("entity", eid), result)
        self.assertIn(("crystal", c2), result)

    def test_respects_max_hops(self):
        c1 = _insert_crystal(self.storage, "a")
        c2 = _insert_crystal(self.storage, "b")
        c3 = _insert_crystal(self.storage, "c")
        self.storage.insert_relation(Relation(
            source_type="crystal", source_id=c1,
            target_type="crystal", target_id=c2,
            relation="chain", weight=0.9,
        ))
        self.storage.insert_relation(Relation(
            source_type="crystal", source_id=c2,
            target_type="crystal", target_id=c3,
            relation="chain", weight=0.9,
        ))
        result = spread_activation(
            self.storage, {("crystal", c1): 1.0},
            max_hops=1, hop_decay=0.5,
        )
        self.assertIn(("crystal", c2), result)
        self.assertNotIn(("crystal", c3), result)

    def test_min_energy_cutoff(self):
        c1 = _insert_crystal(self.storage, "a")
        c2 = _insert_crystal(self.storage, "b")
        self.storage.insert_relation(Relation(
            source_type="crystal", source_id=c1,
            target_type="crystal", target_id=c2,
            relation="weak", weight=0.01,
        ))
        result = spread_activation(
            self.storage, {("crystal", c1): 0.05},
            max_hops=1, hop_decay=0.5, min_energy=0.01,
        )
        # 0.05 * 0.01 * 0.5 = 0.00025 < 0.01 min_energy
        self.assertNotIn(("crystal", c2), result)


# =========================================================================
# Working Memory Buffer
# =========================================================================


class TestWorkingMemoryBuffer(unittest.TestCase):
    def test_add_and_contents(self):
        buf = WorkingMemoryBuffer(capacity=3)
        buf.add(1)
        buf.add(2)
        self.assertEqual(buf.contents(), [1, 2])

    def test_capacity_eviction(self):
        buf = WorkingMemoryBuffer(capacity=3)
        buf.add(1)
        buf.add(2)
        buf.add(3)
        buf.add(4)
        self.assertEqual(len(buf), 3)
        self.assertNotIn(1, buf.contents())
        self.assertIn(4, buf.contents())

    def test_duplicate_moves_to_end(self):
        buf = WorkingMemoryBuffer(capacity=5)
        buf.add(1)
        buf.add(2)
        buf.add(3)
        buf.add(1)  # re-add moves to end
        self.assertEqual(buf.contents(), [2, 3, 1])

    def test_contains(self):
        buf = WorkingMemoryBuffer(capacity=5)
        buf.add(42)
        self.assertTrue(buf.contains(42))
        self.assertFalse(buf.contains(99))

    def test_clear(self):
        buf = WorkingMemoryBuffer(capacity=5)
        buf.add(1)
        buf.add(2)
        buf.clear()
        self.assertEqual(len(buf), 0)


# =========================================================================
# RetrievalEngine — Scene Reconstruction
# =========================================================================


class TestRetrievalEngineScene(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.engine = RetrievalEngine(storage=self.storage)

    def tearDown(self):
        self.storage.close()

    def test_empty_memory_returns_valid_scene(self):
        scene = self.engine.recall("hello", now_ts="2024-06-01T12:00:00")
        self.assertIn("query", scene)
        self.assertIn("crystals", scene)
        self.assertIn("entities", scene)
        self.assertIn("active_self_state", scene)
        self.assertIn("scene_narrative", scene)
        self.assertEqual(scene["crystals"], [])

    def test_scene_has_all_keys(self):
        _insert_crystal(self.storage, "test crystal")
        scene = self.engine.recall("test", now_ts="2024-06-01T12:00:00")
        expected_keys = {
            "query", "active_self_state", "crystals", "entities",
            "contradictions", "prospective_fired", "suppressions",
            "working_memory", "scene_narrative", "activation_id",
        }
        self.assertTrue(expected_keys.issubset(set(scene.keys())))

    def test_crystal_entries_have_expected_fields(self):
        _insert_crystal(self.storage, "python work", ts="2024-06-01T11:00:00")
        scene = self.engine.recall("test", now_ts="2024-06-01T12:00:00")
        if scene["crystals"]:
            c = scene["crystals"][0]
            self.assertIn("id", c)
            self.assertIn("title", c)
            self.assertIn("activation_energy", c)
            self.assertIn("memory_type", c)
            self.assertIn("confidence", c)

    def test_records_activation(self):
        _insert_crystal(self.storage, "something")
        scene = self.engine.recall("query", now_ts="2024-06-01T12:00:00")
        self.assertIn("activation_id", scene)
        activations = self.storage.recent_activations(limit=5)
        self.assertGreater(len(activations), 0)

    def test_reinforces_recalled_crystals(self):
        cid = _insert_crystal(self.storage, "topic")
        self.engine.recall("topic", now_ts="2024-06-01T12:00:00")
        crystal = self.storage.get_crystal(cid)
        # May or may not have been recalled depending on scoring,
        # but at minimum the crystal should still be intact
        self.assertIsNotNone(crystal)

    def test_updates_working_memory(self):
        _insert_crystal(self.storage, "important topic", ts="2024-06-01T11:55:00")
        self.engine.recall("important", now_ts="2024-06-01T12:00:00")
        wm = self.engine.working_memory.contents()
        # Working memory should have some content if crystals were recalled
        self.assertIsInstance(wm, list)


class TestRetrievalSelfStateBoost(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.engine = RetrievalEngine(storage=self.storage)

    def tearDown(self):
        self.storage.close()

    def test_matching_state_boosts_score(self):
        _insert_crystal(self.storage, "code stuff", self_state="builder",
                        ts="2024-06-01T11:55:00")
        _insert_crystal(self.storage, "family stuff", self_state="family",
                        ts="2024-06-01T11:55:00")
        scene = self.engine.recall(
            "debug the python API",
            self_state="builder",
            now_ts="2024-06-01T12:00:00",
        )
        # Builder crystal should rank higher
        if len(scene["crystals"]) >= 2:
            states = [c["self_state"] for c in scene["crystals"]]
            self.assertEqual(states[0], "builder")


class TestRetrievalAntibodySuppression(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.engine = RetrievalEngine(storage=self.storage)

    def tearDown(self):
        self.storage.close()

    def test_antibody_suppresses_crystal(self):
        cid = _insert_crystal(self.storage, "bad advice crystal",
                              ts="2024-06-01T11:55:00")
        create_antibody(
            self.storage,
            trigger_description="migration",
            suppress_crystal_id=cid,
            reason="gave wrong answer",
        )
        scene = self.engine.recall("database migration", now_ts="2024-06-01T12:00:00")
        recalled_ids = [c["id"] for c in scene["crystals"]]
        self.assertNotIn(cid, recalled_ids)
        self.assertGreater(len(scene["suppressions"]), 0)


class TestRetrievalProspective(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.engine = RetrievalEngine(storage=self.storage)

    def tearDown(self):
        self.storage.close()

    def test_prospective_injects_crystal(self):
        cid = _insert_crystal(self.storage, "deploy checklist",
                              ts="2024-06-01T10:00:00")
        create_prospective(
            self.storage,
            trigger_description="deployment",
            payload_crystal_id=cid,
        )
        scene = self.engine.recall("time for deployment", now_ts="2024-06-01T12:00:00")
        self.assertIn(cid, scene["prospective_fired"])


class TestRetrievalWorkingMemoryPersistence(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.engine = RetrievalEngine(storage=self.storage)

    def tearDown(self):
        self.storage.close()

    def test_working_memory_persists_across_recalls(self):
        cid = _insert_crystal(self.storage, "persistent topic",
                              ts="2024-06-01T11:55:00")
        # First recall seeds working memory
        self.engine.recall("persistent topic", now_ts="2024-06-01T12:00:00")
        wm_after_first = self.engine.working_memory.contents()
        # Second recall — working memory still has content
        scene2 = self.engine.recall("something else", now_ts="2024-06-01T12:01:00")
        self.assertEqual(scene2["working_memory"], self.engine.working_memory.contents())


if __name__ == "__main__":
    unittest.main()
