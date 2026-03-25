"""End-to-end integration test — simulates multi-day memory lifecycle.

No Ollama required. Runs entirely offline using keyword/structural scoring.
"""

import time
import unittest

from semantic_gravity_memory import Memory


class TestFullLifecycle(unittest.TestCase):
    """Simulate 5 days of conversations across multiple self-states."""

    def setUp(self):
        self.memory = Memory(db_path=":memory:")

    def tearDown(self):
        self.memory.close()

    # ----- Day 1: Builder work -----------------------------------------------

    def _day1(self):
        self.memory.ingest(
            "I need to build a Python API with Flask and deploy to Docker",
            actor="user", kind="chat_message",
        )
        self.memory.ingest(
            "The database will use SQLite for now, maybe Postgres later",
            actor="user", kind="chat_message",
        )
        self.memory.ingest(
            "I always prefer single-file apps for prototypes",
            actor="user", kind="chat_message",
        )

    # ----- Day 2: Founder work -----------------------------------------------

    def _day2(self):
        self.memory.ingest(
            "The client invoice for South Bay Solutions needs to go out today",
            actor="user", kind="chat_message",
        )
        self.memory.ingest(
            "I like JavaScript for frontend work",
            actor="user", kind="chat_message",
        )

    # ----- Day 3: Contradiction + prospective --------------------------------

    def _day3(self):
        self.memory.ingest(
            "I hate JavaScript, it's terrible for anything serious",
            actor="user", kind="chat_message",
        )
        # Set a prospective memory
        crystals = self.memory._storage.recent_crystals(limit=1)
        if crystals:
            self.memory.set_prospective(
                "deployment", crystals[0].id,
            )

    # ----- Day 4: More builder work ------------------------------------------

    def _day4(self):
        for i in range(5):
            self.memory.ingest(
                f"Debugging the Flask API endpoint, attempt {i+1}",
                actor="user", kind="chat_message",
            )

    # ----- Day 5: Recall + consolidation -------------------------------------

    def _day5(self):
        self.memory.ingest(
            "Time for deployment of the Flask app",
            actor="user", kind="chat_message",
        )

    # =========================================================================
    # Tests
    # =========================================================================

    def test_crystals_created(self):
        self._day1()
        stats = self.memory.stats()
        self.assertGreaterEqual(stats["active_crystals"], 3)

    def test_entities_extracted(self):
        self._day1()
        stats = self.memory.stats()
        self.assertGreater(stats["total_entities"], 0)

    def test_semantic_memory_classified(self):
        self._day1()
        # "I always prefer" should be semantic
        crystals = self.memory._storage.all_crystals()
        semantic = [c for c in crystals if c.memory_type == "semantic"]
        self.assertGreater(len(semantic), 0)

    def test_self_state_detection(self):
        self._day1()
        crystals = self.memory._storage.all_crystals()
        builder = [c for c in crystals if c.self_state == "builder"]
        self.assertGreater(len(builder), 0)

    def test_contradiction_detected(self):
        self._day2()
        self._day3()
        contras = self.memory._storage.open_contradictions()
        self.assertGreater(len(contras), 0)

    def test_recall_returns_scene(self):
        self._day1()
        self._day2()
        scene = self.memory.recall("what tools am I using?")
        self.assertIn("crystals", scene)
        self.assertIn("active_self_state", scene)
        self.assertIn("scene_narrative", scene)
        self.assertIn("activation_id", scene)

    def test_recall_records_activation(self):
        self._day1()
        self.memory.recall("Python API")
        activations = self.memory._storage.recent_activations(limit=5)
        self.assertGreater(len(activations), 0)

    def test_prospective_memory_fires(self):
        self._day1()
        self._day2()
        self._day3()
        self._day5()  # mentions "deployment"
        scene = self.memory.recall("deployment")
        self.assertGreater(len(scene.get("prospective_fired", [])), 0)

    def test_antibody_suppresses(self):
        self._day1()
        crystals = self.memory._storage.recent_crystals(limit=1)
        cid = crystals[0].id
        self.memory.suppress(cid, reason="bad advice", trigger="Python")
        scene = self.memory.recall("Python", now_ts="2024-06-01T12:00:00")
        recalled_ids = [c["id"] for c in scene["crystals"]]
        self.assertNotIn(cid, recalled_ids)

    def test_consolidation_runs(self):
        self._day1()
        self._day2()
        self._day3()
        self._day4()
        log = self.memory.consolidate()
        self.assertIn("decay", log)
        self.assertIn("merge", log)
        self.assertIn("schema", log)
        self.assertIn("graduation", log)
        self.assertIn("carrying_capacity", log)

    def test_contradiction_resolved_by_consolidation(self):
        self._day2()
        self._day3()
        self.assertGreater(len(self.memory._storage.open_contradictions()), 0)
        self.memory.consolidate()
        self.assertEqual(len(self.memory._storage.open_contradictions()), 0)

    def test_answer_with_mock_llm(self):
        self._day1()
        answer, scene = self.memory.answer(
            "what framework am I using?",
            chat_fn=lambda prompt: "Based on memory, you're using Flask.",
        )
        self.assertEqual(answer, "Based on memory, you're using Flask.")
        self.assertIn("crystals", scene)
        # Verify the assistant response was ingested
        events = self.memory._storage.recent_events(limit=5)
        assistant = [e for e in events if e.actor == "assistant"]
        self.assertGreater(len(assistant), 0)

    def test_feedback_updates_confidence(self):
        self._day1()
        scene = self.memory.recall("Python")
        act_id = scene["activation_id"]
        self.memory.feedback(act_id, 0.9, "builder")
        conf = self.memory._engine.metamemory.domain_confidence("builder")
        self.assertAlmostEqual(conf, 0.9)

    def test_export_contains_all_data(self):
        self._day1()
        self._day2()
        export = self.memory.export()
        self.assertIn("events", export)
        self.assertIn("crystals", export)
        self.assertIn("entities", export)
        self.assertIn("relations", export)
        self.assertGreater(len(export["events"]), 0)
        self.assertGreater(len(export["crystals"]), 0)

    def test_stats_comprehensive(self):
        self._day1()
        self._day2()
        self._day3()
        self._day4()
        self.memory.consolidate()
        s = self.memory.stats()
        self.assertIn("total_crystals", s)
        self.assertIn("active_crystals", s)
        self.assertIn("semantic_crystals", s)
        self.assertIn("episodic_crystals", s)
        self.assertIn("total_entities", s)
        self.assertIn("schemas", s)
        self.assertIn("domain_confidences", s)
        self.assertIn("consolidation_log", s)

    def test_memory_repr(self):
        self._day1()
        r = repr(self.memory)
        self.assertIn("Memory", r)
        self.assertIn("crystals=", r)

    def test_context_manager(self):
        with Memory(db_path=":memory:") as mem:
            mem.ingest("test message")
            s = mem.stats()
            self.assertGreaterEqual(s["active_crystals"], 1)


class TestPerformance(unittest.TestCase):
    """Ensure the system handles moderate scale without blowing up."""

    def test_hundred_ingestions(self):
        with Memory(db_path=":memory:") as mem:
            for i in range(100):
                mem.ingest(f"Message number {i} about Python Flask Docker Postgres")
            stats = mem.stats()
            self.assertGreaterEqual(stats["active_crystals"], 50)

    def test_recall_after_hundred(self):
        with Memory(db_path=":memory:") as mem:
            for i in range(100):
                mem.ingest(f"Building feature {i} with Python and Flask")
            start = time.time()
            scene = mem.recall("what have I been building?")
            elapsed = time.time() - start
            self.assertIn("crystals", scene)
            # Should complete in under 2 seconds even without embeddings
            self.assertLess(elapsed, 2.0)

    def test_consolidation_after_hundred(self):
        with Memory(db_path=":memory:") as mem:
            for i in range(100):
                mem.ingest(f"Debug session {i} for the API endpoint")
            log = mem.consolidate()
            self.assertIn("decay", log)


if __name__ == "__main__":
    unittest.main()
