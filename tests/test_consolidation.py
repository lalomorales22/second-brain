"""Tests for consolidation passes and the MemoryEngine orchestrator."""

import time
import unittest

from semantic_gravity_memory.models import (
    Contradiction,
    Crystal,
    Entity,
    Event,
    SalienceVector,
)
from semantic_gravity_memory.storage.sqlite_backend import SQLiteBackend
from semantic_gravity_memory.core.consolidation import (
    Consolidator,
    ConsolidationDaemon,
)
from semantic_gravity_memory.core.engine import MemoryEngine


def _sal(**kw):
    defaults = dict(emotional=0.5, practical=0.5, identity=0.5,
                    temporal=0.2, uncertainty=0.3, novelty=0.3)
    defaults.update(kw)
    return SalienceVector(**defaults)


def _crystal(storage, title, ts="2024-06-01T12:00:00", embedding=None, **kw):
    defaults = dict(
        title=title, theme=title, summary=f"about {title}",
        created_ts=ts, salience=_sal(), confidence=0.8,
        decay_rate=0.05, memory_type="episodic",
        embedding=embedding,
    )
    defaults.update(kw)
    cid = storage.insert_crystal(Crystal(**defaults))
    return storage.get_crystal(cid)


# =========================================================================
# Decay pass
# =========================================================================

class TestDecayPass(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")
        self.con = Consolidator(self.db)

    def tearDown(self):
        self.db.close()

    def test_weak_crystal_goes_dormant(self):
        _crystal(self.db, "ephemeral", ts="2020-01-01T00:00:00",
                 decay_rate=1.0, confidence=0.2,
                 salience=SalienceVector(practical=0.05))
        log = self.con.run_pass(now_ts="2024-06-01T00:00:00")
        self.assertGreaterEqual(log["decay"]["dormant"], 1)

    def test_strong_crystal_survives(self):
        _crystal(self.db, "strong", ts="2024-06-01T11:00:00",
                 decay_rate=0.01)
        log = self.con.run_pass(now_ts="2024-06-01T12:00:00")
        self.assertEqual(log["decay"]["dormant"], 0)


# =========================================================================
# Merge pass
# =========================================================================

class TestMergePass(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")
        self.con = Consolidator(self.db, merge_threshold=0.85)

    def tearDown(self):
        self.db.close()

    def test_similar_crystals_merged(self):
        c1 = _crystal(self.db, "alpha", embedding=[1.0, 0.0, 0.0],
                       ts="2024-06-01T12:00:00")
        c2 = _crystal(self.db, "beta", embedding=[0.99, 0.01, 0.0],
                       ts="2024-06-01T12:01:00")
        log = self.con.run_pass(now_ts="2024-06-01T12:05:00")
        self.assertEqual(log["merge"]["merged"], 1)

        # Keeper should still be active
        refreshed1 = self.db.get_crystal(c1.id)
        self.assertIsNone(refreshed1.valid_to_ts)

        # Absorbed should be dormant with parent link
        refreshed2 = self.db.get_crystal(c2.id)
        self.assertIsNotNone(refreshed2.valid_to_ts)
        self.assertEqual(refreshed2.parent_crystal_id, c1.id)

    def test_dissimilar_crystals_not_merged(self):
        _crystal(self.db, "north", embedding=[1.0, 0.0, 0.0],
                 ts="2024-06-01T12:00:00")
        _crystal(self.db, "south", embedding=[0.0, 0.0, 1.0],
                 ts="2024-06-01T12:01:00")
        log = self.con.run_pass(now_ts="2024-06-01T12:05:00")
        self.assertEqual(log["merge"]["merged"], 0)

    def test_merge_combines_events_and_entities(self):
        c1 = _crystal(self.db, "a", embedding=[1.0, 0.0, 0.0],
                       source_event_ids=[1, 2], entity_ids=[10, 11])
        c2 = _crystal(self.db, "b", embedding=[0.99, 0.01, 0.0],
                       source_event_ids=[3], entity_ids=[11, 12])
        self.con.run_pass(now_ts="2024-06-01T12:05:00")

        keeper = self.db.get_crystal(c1.id)
        self.assertEqual(set(keeper.source_event_ids), {1, 2, 3})
        self.assertEqual(set(keeper.entity_ids), {10, 11, 12})

    def test_merge_increments_version(self):
        c1 = _crystal(self.db, "a", embedding=[1.0, 0.0, 0.0])
        _crystal(self.db, "b", embedding=[0.99, 0.01, 0.0])
        self.con.run_pass(now_ts="2024-06-01T12:05:00")
        self.assertEqual(self.db.get_crystal(c1.id).version, 2)


# =========================================================================
# Schema extraction
# =========================================================================

class TestSchemaExtraction(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")
        self.con = Consolidator(self.db)

    def tearDown(self):
        self.db.close()

    def test_creates_schema_from_recurring_pattern(self):
        shared = [1, 2]
        for i in range(4):
            _crystal(self.db, f"build_{i}", self_state="builder",
                     entity_ids=shared + [10 + i],
                     ts=f"2024-06-01T{12+i}:00:00")
        log = self.con.run_pass(now_ts="2024-06-01T18:00:00")
        self.assertGreaterEqual(log["schema"]["extracted"], 1)
        schemas = self.db.all_schemas()
        self.assertTrue(any("builder" in s.name for s in schemas))

    def test_no_schema_with_few_crystals(self):
        _crystal(self.db, "lonely", self_state="researcher", entity_ids=[1])
        log = self.con.run_pass(now_ts="2024-06-01T12:05:00")
        self.assertEqual(log["schema"]["extracted"], 0)

    def test_schema_updates_on_rerun(self):
        shared = [1, 2]
        for i in range(3):
            _crystal(self.db, f"round1_{i}", self_state="builder",
                     entity_ids=shared, ts=f"2024-06-01T{10+i}:00:00")
        self.con.run_pass(now_ts="2024-06-01T15:00:00")
        # Add more crystals and re-run
        for i in range(2):
            _crystal(self.db, f"round2_{i}", self_state="builder",
                     entity_ids=shared, ts=f"2024-06-02T{10+i}:00:00")
        self.con.run_pass(now_ts="2024-06-02T15:00:00")
        # Should still be 1 schema, just updated
        schemas = self.db.all_schemas()
        builder_schemas = [s for s in schemas if "builder" in s.name]
        self.assertEqual(len(builder_schemas), 1)
        self.assertGreater(builder_schemas[0].activation_count, 1)


# =========================================================================
# Contradiction resolution
# =========================================================================

class TestContradictionResolution(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")
        self.con = Consolidator(self.db)

    def tearDown(self):
        self.db.close()

    def test_resolves_open_contradictions(self):
        eid1 = self.db.insert_event(Event(
            ts="2024-01-01T00:00:00", actor="user", kind="chat", content="old"))
        eid2 = self.db.insert_event(Event(
            ts="2024-06-01T00:00:00", actor="user", kind="chat", content="new"))
        self.db.insert_contradiction(Contradiction(
            topic="test", claim_a="old", claim_b="new",
            evidence_event_a=eid1, evidence_event_b=eid2))

        log = self.con.run_pass(now_ts="2024-06-01T12:00:00")
        self.assertEqual(log["contradictions"]["resolved"], 1)
        self.assertEqual(len(self.db.open_contradictions()), 0)

    def test_newer_claim_wins(self):
        eid1 = self.db.insert_event(Event(
            ts="2024-01-01T00:00:00", actor="user", kind="chat", content="old"))
        eid2 = self.db.insert_event(Event(
            ts="2024-06-01T00:00:00", actor="user", kind="chat", content="new"))
        self.db.insert_contradiction(Contradiction(
            topic="test", claim_a="old", claim_b="new",
            evidence_event_a=eid1, evidence_event_b=eid2))

        self.con.run_pass(now_ts="2024-06-01T12:00:00")
        all_c = self.db.all_contradictions()
        self.assertEqual(all_c[0].resolution_state, "resolved_b")


# =========================================================================
# Graduation (episodic → semantic)
# =========================================================================

class TestGraduation(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")
        self.con = Consolidator(self.db, graduation_access_min=5)

    def tearDown(self):
        self.db.close()

    def test_high_access_low_temporal_graduates(self):
        c = _crystal(self.db, "wisdom", memory_type="episodic",
                     access_count=10,
                     salience=SalienceVector(temporal=0.1, practical=0.8))
        log = self.con.run_pass(now_ts="2024-06-01T12:00:00")
        self.assertEqual(log["graduation"]["graduated"], 1)
        refreshed = self.db.get_crystal(c.id)
        self.assertEqual(refreshed.memory_type, "semantic")
        self.assertLess(refreshed.decay_rate, 0.05)

    def test_low_access_stays_episodic(self):
        _crystal(self.db, "fleeting", memory_type="episodic",
                 access_count=2)
        log = self.con.run_pass(now_ts="2024-06-01T12:00:00")
        self.assertEqual(log["graduation"]["graduated"], 0)

    def test_high_temporal_stays_episodic(self):
        _crystal(self.db, "timely", memory_type="episodic",
                 access_count=10,
                 salience=SalienceVector(temporal=0.8, practical=0.8))
        log = self.con.run_pass(now_ts="2024-06-01T12:00:00")
        self.assertEqual(log["graduation"]["graduated"], 0)

    def test_already_semantic_not_counted(self):
        _crystal(self.db, "known", memory_type="semantic",
                 access_count=10)
        log = self.con.run_pass(now_ts="2024-06-01T12:00:00")
        self.assertEqual(log["graduation"]["graduated"], 0)


# =========================================================================
# Carrying capacity
# =========================================================================

class TestCarryingCapacity(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")
        self.con = Consolidator(self.db, carrying_capacity=3)

    def tearDown(self):
        self.db.close()

    def test_evicts_weakest_when_over_budget(self):
        for i in range(5):
            _crystal(self.db, f"c{i}",
                     ts=f"2024-06-01T{10+i}:00:00",
                     confidence=0.1 + i * 0.2)
        log = self.con.run_pass(now_ts="2024-06-01T18:00:00")
        self.assertGreaterEqual(log["carrying_capacity"]["evicted"], 2)
        active = [c for c in self.db.all_crystals() if not c.valid_to_ts]
        self.assertLessEqual(len(active), 3)

    def test_no_eviction_under_budget(self):
        for i in range(2):
            _crystal(self.db, f"c{i}")
        log = self.con.run_pass(now_ts="2024-06-01T12:05:00")
        self.assertEqual(log["carrying_capacity"]["evicted"], 0)


# =========================================================================
# Full consolidation pass
# =========================================================================

class TestFullPass(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")
        self.con = Consolidator(self.db)

    def tearDown(self):
        self.db.close()

    def test_log_has_all_keys(self):
        log = self.con.run_pass(now_ts="2024-06-01T12:00:00")
        expected = {"ts", "decay", "merge", "schema", "contradictions",
                    "graduation", "carrying_capacity", "clustering"}
        self.assertTrue(expected.issubset(set(log.keys())))

    def test_log_persisted(self):
        self.con.run_pass(now_ts="2024-06-01T12:00:00")
        self.con.run_pass(now_ts="2024-06-01T13:00:00")
        history = self.con.get_log()
        self.assertEqual(len(history), 2)


# =========================================================================
# ConsolidationDaemon
# =========================================================================

class TestDaemon(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")
        self.con = Consolidator(self.db)

    def tearDown(self):
        self.db.close()

    def test_start_and_stop(self):
        daemon = ConsolidationDaemon(self.con, heartbeat_seconds=0.05)
        daemon.start()
        self.assertTrue(daemon.running)
        time.sleep(0.15)  # let at least one pass run
        daemon.stop(timeout=2.0)
        self.assertFalse(daemon.running)
        # Should have at least 1 log entry
        history = self.con.get_log()
        self.assertGreaterEqual(len(history), 1)

    def test_double_start_safe(self):
        daemon = ConsolidationDaemon(self.con, heartbeat_seconds=60)
        daemon.start()
        daemon.start()  # should not crash or create duplicate threads
        daemon.stop(timeout=1.0)


# =========================================================================
# MemoryEngine
# =========================================================================

class TestMemoryEngine(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")
        self.engine = MemoryEngine(self.db)

    def tearDown(self):
        self.engine.stop_daemon()
        self.db.close()

    def test_ingest_and_recall(self):
        self.engine.ingest("I prefer Python for backend work")
        scene = self.engine.recall("what language?", now_ts="2024-06-01T12:00:00")
        self.assertIn("crystals", scene)
        self.assertIn("query", scene)

    def test_answer_with_mock_chat(self):
        self.engine.ingest("I use Flask for APIs")
        answer, scene = self.engine.answer(
            "what framework?",
            chat_fn=lambda prompt: "You mentioned Flask.",
            now_ts="2024-06-01T12:00:00",
        )
        self.assertEqual(answer, "You mentioned Flask.")
        self.assertIn("crystals", scene)

    def test_answer_ingests_response(self):
        self.engine.ingest("hello")
        self.engine.answer(
            "hi", chat_fn=lambda p: "hey there",
            now_ts="2024-06-01T12:00:00",
        )
        events = self.db.recent_events(limit=10)
        assistant_events = [e for e in events if e.actor == "assistant"]
        self.assertGreater(len(assistant_events), 0)

    def test_consolidate(self):
        self.engine.ingest("something")
        log = self.engine.consolidate(now_ts="2024-06-01T12:00:00")
        self.assertIn("decay", log)

    def test_feedback(self):
        self.engine.ingest("test")
        scene = self.engine.recall("test", now_ts="2024-06-01T12:00:00")
        act_id = scene.get("activation_id", 1)
        self.engine.feedback(act_id, 0.9, "builder")
        conf = self.engine.metamemory.domain_confidence("builder")
        self.assertAlmostEqual(conf, 0.9)

    def test_set_prospective(self):
        eid, cid = self.engine.ingest("deployment checklist")
        pid = self.engine.set_prospective("deployment", cid)
        self.assertIsInstance(pid, int)
        active = self.db.active_prospective_memories()
        self.assertEqual(len(active), 1)

    def test_suppress(self):
        _, cid = self.engine.ingest("bad advice")
        abid = self.engine.suppress(cid, reason="wrong answer")
        self.assertIsInstance(abid, int)
        self.assertEqual(len(self.db.active_antibodies()), 1)

    def test_export(self):
        self.engine.ingest("data")
        export = self.engine.export()
        self.assertIn("events", export)
        self.assertIn("crystals", export)

    def test_stats(self):
        self.engine.ingest("one")
        self.engine.ingest("two")
        s = self.engine.stats()
        self.assertGreaterEqual(s["total_crystals"], 2)
        self.assertIn("active_crystals", s)
        self.assertIn("domain_confidences", s)
        self.assertIn("schemas", s)

    def test_daemon_lifecycle(self):
        self.engine.start_daemon(heartbeat_seconds=60)
        self.assertTrue(self.engine.daemon_running)
        self.engine.stop_daemon()
        self.assertFalse(self.engine.daemon_running)


if __name__ == "__main__":
    unittest.main()
