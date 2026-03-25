"""Tests for the crystal forge pipeline."""

import unittest

from semantic_gravity_memory.core.crystal_forge import CrystalForge
from semantic_gravity_memory.models import SalienceVector
from semantic_gravity_memory.storage.sqlite_backend import SQLiteBackend


class TestCrystalForgeBasic(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.forge = CrystalForge(storage=self.storage)

    def tearDown(self):
        self.storage.close()

    def test_ingest_returns_valid_ids(self):
        event_id, crystal_id = self.forge.ingest("I'm building a Python API with Flask")
        self.assertIsInstance(event_id, int)
        self.assertIsInstance(crystal_id, int)
        self.assertGreater(event_id, 0)
        self.assertGreater(crystal_id, 0)

    def test_event_stored_correctly(self):
        event_id, _ = self.forge.ingest("Hello world")
        event = self.storage.get_event(event_id)
        self.assertIsNotNone(event)
        self.assertEqual(event.content, "Hello world")
        self.assertEqual(event.actor, "user")

    def test_crystal_has_title_and_summary(self):
        _, crystal_id = self.forge.ingest("I'm building a Python API with Flask")
        crystal = self.storage.get_crystal(crystal_id)
        self.assertIsNotNone(crystal)
        self.assertNotEqual(crystal.title, "")
        self.assertNotEqual(crystal.summary, "")

    def test_crystal_has_salience_vector(self):
        _, crystal_id = self.forge.ingest("I need to fix this urgent bug")
        crystal = self.storage.get_crystal(crystal_id)
        self.assertIsInstance(crystal.salience, SalienceVector)
        self.assertGreater(crystal.salience.practical, 0.0)

    def test_entities_stored(self):
        self.forge.ingest("Deploy the Flask app to Docker with Postgres")
        entities = self.storage.top_entities(limit=10)
        names = [e.name.lower() for e in entities]
        self.assertTrue(any("flask" in n for n in names))

    def test_entity_ids_linked_to_crystal(self):
        _, crystal_id = self.forge.ingest("I use Python and Flask daily")
        crystal = self.storage.get_crystal(crystal_id)
        self.assertGreater(len(crystal.entity_ids), 0)

    def test_relations_created(self):
        self.forge.ingest("I use Python and Flask together")
        relations = self.storage.all_relations()
        self.assertGreater(len(relations), 0)
        rel_types = {r.relation for r in relations}
        self.assertIn("crystallized_into", rel_types)
        self.assertIn("mentions", rel_types)

    def test_valid_from_ts_set(self):
        _, crystal_id = self.forge.ingest("test content")
        crystal = self.storage.get_crystal(crystal_id)
        self.assertIsNotNone(crystal.valid_from_ts)

    def test_compressed_narrative_populated(self):
        _, crystal_id = self.forge.ingest("Building a memory system in Python")
        crystal = self.storage.get_crystal(crystal_id)
        self.assertNotEqual(crystal.compressed_narrative, "")


class TestCrystalForgeActors(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.forge = CrystalForge(storage=self.storage)

    def tearDown(self):
        self.storage.close()

    def test_user_highest_confidence(self):
        _, cid = self.forge.ingest("I prefer Python", actor="user")
        crystal = self.storage.get_crystal(cid)
        self.assertGreater(crystal.confidence, 0.7)

    def test_assistant_medium_confidence(self):
        _, cid = self.forge.ingest("Python is popular", actor="assistant")
        crystal = self.storage.get_crystal(cid)
        self.assertLess(crystal.confidence, 0.7)
        self.assertGreater(crystal.confidence, 0.5)

    def test_system_lowest_confidence(self):
        _, cid = self.forge.ingest("System observation", actor="system")
        crystal = self.storage.get_crystal(cid)
        self.assertLessEqual(crystal.confidence, 0.55)

    def test_event_actor_stored(self):
        eid, _ = self.forge.ingest("test", actor="assistant", kind="response")
        event = self.storage.get_event(eid)
        self.assertEqual(event.actor, "assistant")
        self.assertEqual(event.kind, "response")


class TestCrystalForgeMemoryType(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.forge = CrystalForge(storage=self.storage)

    def tearDown(self):
        self.storage.close()

    def test_episodic_by_default(self):
        _, cid = self.forge.ingest("I deployed the app today")
        crystal = self.storage.get_crystal(cid)
        self.assertEqual(crystal.memory_type, "episodic")

    def test_semantic_for_always_statements(self):
        _, cid = self.forge.ingest("I always prefer single-file Python apps")
        crystal = self.storage.get_crystal(cid)
        self.assertEqual(crystal.memory_type, "semantic")

    def test_semantic_for_beliefs(self):
        _, cid = self.forge.ingest("I believe in local-first architecture")
        crystal = self.storage.get_crystal(cid)
        self.assertEqual(crystal.memory_type, "semantic")

    def test_semantic_for_preferences(self):
        _, cid = self.forge.ingest("I prefer dark mode in all my editors")
        crystal = self.storage.get_crystal(cid)
        self.assertEqual(crystal.memory_type, "semantic")


class TestCrystalForgeContradictions(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.forge = CrystalForge(storage=self.storage)

    def tearDown(self):
        self.storage.close()

    def test_preference_contradiction_detected(self):
        self.forge.ingest("I like JavaScript for frontend work")
        self.forge.ingest("I hate JavaScript, it's terrible")
        contras = self.storage.open_contradictions()
        self.assertGreater(len(contras), 0)

    def test_contradiction_marks_crystal_tension(self):
        self.forge.ingest("I like JavaScript for frontend work")
        _, cid = self.forge.ingest("I hate JavaScript, it's terrible")
        crystal = self.storage.get_crystal(cid)
        self.assertEqual(crystal.contradiction_state, "tension")

    def test_no_false_positive_different_subjects(self):
        self.forge.ingest("I like Python for backend")
        self.forge.ingest("I hate JavaScript for frontend")
        contras = self.storage.open_contradictions()
        self.assertEqual(len(contras), 0)


class TestCrystalForgeSelfState(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.forge = CrystalForge(storage=self.storage)

    def tearDown(self):
        self.storage.close()

    def test_builder_state_detected(self):
        _, cid = self.forge.ingest("I need to debug this Python API endpoint")
        crystal = self.storage.get_crystal(cid)
        self.assertEqual(crystal.self_state, "builder")

    def test_founder_state_detected(self):
        _, cid = self.forge.ingest(
            "The client invoice needs to be sent for the contract"
        )
        crystal = self.storage.get_crystal(cid)
        self.assertEqual(crystal.self_state, "founder")

    def test_general_for_ambiguous_text(self):
        _, cid = self.forge.ingest("The weather looks nice today")
        crystal = self.storage.get_crystal(cid)
        self.assertEqual(crystal.self_state, "general")


class TestCrystalForgeDecayRate(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.forge = CrystalForge(storage=self.storage)

    def tearDown(self):
        self.storage.close()

    def test_user_decays_slower_than_assistant(self):
        _, cid_user = self.forge.ingest("test content", actor="user")
        _, cid_asst = self.forge.ingest("test content", actor="assistant")
        user_crystal = self.storage.get_crystal(cid_user)
        asst_crystal = self.storage.get_crystal(cid_asst)
        self.assertLess(user_crystal.decay_rate, asst_crystal.decay_rate)

    def test_decay_rate_positive(self):
        _, cid = self.forge.ingest("anything")
        crystal = self.storage.get_crystal(cid)
        self.assertGreater(crystal.decay_rate, 0.0)


class TestCrystalForgeAccumulation(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.forge = CrystalForge(storage=self.storage)

    def tearDown(self):
        self.storage.close()

    def test_entity_salience_accumulates(self):
        self.forge.ingest("I use Python for everything")
        self.forge.ingest("Python is great for prototyping")
        self.forge.ingest("My Python scripts are getting complex")
        entity = self.storage.get_entity_by_name("python")
        if entity is None:
            # might be stored with different casing
            entities = self.storage.top_entities(limit=20)
            entity = next((e for e in entities if "python" in e.name.lower()), None)
        self.assertIsNotNone(entity)
        self.assertGreater(entity.mention_count, 1)

    def test_multiple_crystals_created(self):
        self.forge.ingest("First message about Python")
        self.forge.ingest("Second message about Flask")
        self.forge.ingest("Third message about deployment")
        crystals = self.storage.all_crystals()
        self.assertEqual(len(crystals), 3)

    def test_co_occurrence_relations(self):
        self.forge.ingest("Deploy Flask to Docker with Postgres")
        relations = self.storage.all_relations()
        co_rels = [r for r in relations if r.relation == "co_occurred"]
        self.assertGreater(len(co_rels), 0)


class TestCrystalForgeFutureImplications(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.forge = CrystalForge(storage=self.storage)

    def tearDown(self):
        self.storage.close()

    def test_implementation_implication(self):
        _, cid = self.forge.ingest("I need to build a new prototype app")
        crystal = self.storage.get_crystal(cid)
        self.assertIn("implementation", crystal.future_implications)

    def test_question_implication(self):
        _, cid = self.forge.ingest("How does the auth middleware work?")
        crystal = self.storage.get_crystal(cid)
        self.assertIn("question", crystal.future_implications)

    def test_unresolved_for_question(self):
        _, cid = self.forge.ingest("Should we use Redis or Memcached?")
        crystal = self.storage.get_crystal(cid)
        self.assertNotEqual(crystal.unresolved, "")


if __name__ == "__main__":
    unittest.main()
