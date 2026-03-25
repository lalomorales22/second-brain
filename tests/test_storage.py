"""Tests for SQLite storage backend — full CRUD for all entity types."""

import unittest
from semantic_gravity_memory.models import (
    Activation,
    AntibodyMemory,
    Contradiction,
    Crystal,
    Entity,
    Event,
    ProspectiveMemory,
    Relation,
    SalienceVector,
    Schema,
)
from semantic_gravity_memory.storage.sqlite_backend import SQLiteBackend


class TestSQLiteBackendMeta(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")

    def tearDown(self):
        self.db.close()

    def test_set_and_get(self):
        self.db.set_meta("version", "0.1.0")
        self.assertEqual(self.db.get_meta("version"), "0.1.0")

    def test_get_default(self):
        self.assertEqual(self.db.get_meta("missing", "fallback"), "fallback")

    def test_upsert(self):
        self.db.set_meta("k", "v1")
        self.db.set_meta("k", "v2")
        self.assertEqual(self.db.get_meta("k"), "v2")


class TestSQLiteBackendEvents(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")

    def tearDown(self):
        self.db.close()

    def test_insert_and_get(self):
        e = Event(ts="2024-01-01T00:00:00", actor="user", kind="chat", content="hello world", context={"ch": "test"}, salience=0.5)
        eid = self.db.insert_event(e)
        self.assertIsInstance(eid, int)
        self.assertGreater(eid, 0)

        got = self.db.get_event(eid)
        self.assertIsNotNone(got)
        self.assertEqual(got.content, "hello world")
        self.assertEqual(got.actor, "user")
        self.assertEqual(got.context, {"ch": "test"})
        self.assertAlmostEqual(got.salience, 0.5)

    def test_get_missing(self):
        self.assertIsNone(self.db.get_event(9999))

    def test_recent_events_order(self):
        for i in range(5):
            self.db.insert_event(Event(ts=f"2024-01-0{i+1}T00:00:00", actor="user", kind="chat", content=f"msg {i}"))
        recent = self.db.recent_events(limit=3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0].content, "msg 4")  # newest first
        self.assertEqual(recent[2].content, "msg 2")

    def test_embedding_round_trip(self):
        vec = [0.1, 0.2, 0.3, 0.4]
        eid = self.db.insert_event(Event(ts="2024-01-01T00:00:00", actor="user", kind="chat", content="test", embedding=vec))
        got = self.db.get_event(eid)
        self.assertEqual(got.embedding, vec)

    def test_null_embedding(self):
        eid = self.db.insert_event(Event(ts="2024-01-01T00:00:00", actor="user", kind="chat", content="test"))
        got = self.db.get_event(eid)
        self.assertIsNone(got.embedding)

    def test_auto_timestamp(self):
        eid = self.db.insert_event(Event(actor="user", kind="chat", content="no ts"))
        got = self.db.get_event(eid)
        self.assertNotEqual(got.ts, "")


class TestSQLiteBackendEntities(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")

    def tearDown(self):
        self.db.close()

    def test_insert_new(self):
        e = Entity(name="python", kind="tool", salience=0.3)
        eid = self.db.upsert_entity(e)
        self.assertIsInstance(eid, int)
        got = self.db.get_entity(eid)
        self.assertEqual(got.name, "python")
        self.assertEqual(got.kind, "tool")
        self.assertEqual(got.mention_count, 1)

    def test_upsert_existing(self):
        e1 = Entity(name="python", kind="tool", salience=0.3)
        eid1 = self.db.upsert_entity(e1)
        e2 = Entity(name="python", kind="tool", salience=0.2, metadata={"version": "3.12"})
        eid2 = self.db.upsert_entity(e2)
        self.assertEqual(eid1, eid2)  # same entity
        got = self.db.get_entity(eid1)
        self.assertAlmostEqual(got.salience, 0.5)  # 0.3 + 0.2
        self.assertEqual(got.mention_count, 2)
        self.assertEqual(got.metadata["version"], "3.12")

    def test_get_by_name(self):
        self.db.upsert_entity(Entity(name="flask", kind="tool"))
        got = self.db.get_entity_by_name("flask")
        self.assertIsNotNone(got)
        self.assertEqual(got.name, "flask")

    def test_get_by_name_missing(self):
        self.assertIsNone(self.db.get_entity_by_name("nonexistent"))

    def test_top_entities(self):
        self.db.upsert_entity(Entity(name="low", salience=0.1))
        self.db.upsert_entity(Entity(name="high", salience=0.9))
        self.db.upsert_entity(Entity(name="mid", salience=0.5))
        top = self.db.top_entities(limit=2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0].name, "high")


class TestSQLiteBackendCrystals(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")

    def tearDown(self):
        self.db.close()

    def _make_crystal(self, **overrides) -> Crystal:
        defaults = dict(
            title="test crystal",
            theme="testing",
            summary="a test crystal",
            source_event_ids=[1, 2],
            entity_ids=[3],
            salience=SalienceVector(emotional=0.2, practical=0.8, temporal=0.5),
            confidence=0.75,
            self_state="builder",
            memory_type="episodic",
            embedding=[0.1, 0.2, 0.3],
            decay_rate=0.15,
        )
        defaults.update(overrides)
        return Crystal(**defaults)

    def test_insert_and_get(self):
        c = self._make_crystal()
        cid = self.db.insert_crystal(c)
        self.assertIsInstance(cid, int)

        got = self.db.get_crystal(cid)
        self.assertIsNotNone(got)
        self.assertEqual(got.title, "test crystal")
        self.assertEqual(got.source_event_ids, [1, 2])
        self.assertEqual(got.entity_ids, [3])
        self.assertAlmostEqual(got.salience.emotional, 0.2)
        self.assertAlmostEqual(got.salience.practical, 0.8)
        self.assertAlmostEqual(got.salience.temporal, 0.5)
        self.assertAlmostEqual(got.confidence, 0.75)
        self.assertEqual(got.self_state, "builder")
        self.assertEqual(got.memory_type, "episodic")
        self.assertEqual(got.embedding, [0.1, 0.2, 0.3])
        self.assertAlmostEqual(got.decay_rate, 0.15)

    def test_update(self):
        cid = self.db.insert_crystal(self._make_crystal())
        got = self.db.get_crystal(cid)
        got.title = "updated title"
        got.access_count = 5
        got.version = 2
        got.memory_type = "semantic"
        self.db.update_crystal(got)

        refreshed = self.db.get_crystal(cid)
        self.assertEqual(refreshed.title, "updated title")
        self.assertEqual(refreshed.access_count, 5)
        self.assertEqual(refreshed.version, 2)
        self.assertEqual(refreshed.memory_type, "semantic")

    def test_update_without_id_raises(self):
        c = self._make_crystal()
        with self.assertRaises(ValueError):
            self.db.update_crystal(c)

    def test_get_missing(self):
        self.assertIsNone(self.db.get_crystal(9999))

    def test_all_and_recent(self):
        for i in range(5):
            self.db.insert_crystal(self._make_crystal(title=f"crystal {i}"))
        self.assertEqual(len(self.db.all_crystals()), 5)
        recent = self.db.recent_crystals(limit=3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0].title, "crystal 4")  # newest first

    def test_salience_vector_round_trip(self):
        sv = SalienceVector(emotional=0.11, practical=0.22, identity=0.33, temporal=0.44, uncertainty=0.55, novelty=0.66)
        cid = self.db.insert_crystal(self._make_crystal(salience=sv))
        got = self.db.get_crystal(cid)
        self.assertAlmostEqual(got.salience.emotional, 0.11, places=5)
        self.assertAlmostEqual(got.salience.practical, 0.22, places=5)
        self.assertAlmostEqual(got.salience.identity, 0.33, places=5)
        self.assertAlmostEqual(got.salience.temporal, 0.44, places=5)
        self.assertAlmostEqual(got.salience.uncertainty, 0.55, places=5)
        self.assertAlmostEqual(got.salience.novelty, 0.66, places=5)

    def test_parent_crystal_linkage(self):
        cid1 = self.db.insert_crystal(self._make_crystal(title="child"))
        cid2 = self.db.insert_crystal(self._make_crystal(title="parent"))
        child = self.db.get_crystal(cid1)
        child.parent_crystal_id = cid2
        self.db.update_crystal(child)
        refreshed = self.db.get_crystal(cid1)
        self.assertEqual(refreshed.parent_crystal_id, cid2)


class TestSQLiteBackendRelations(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")

    def tearDown(self):
        self.db.close()

    def test_insert_and_query(self):
        r = Relation(source_type="crystal", source_id=1, target_type="entity", target_id=2, relation="mentions", weight=0.7, context={"origin": "extraction"})
        rid = self.db.insert_relation(r)
        self.assertIsInstance(rid, int)

        from_crystal = self.db.relations_from("crystal", 1)
        self.assertEqual(len(from_crystal), 1)
        self.assertEqual(from_crystal[0].relation, "mentions")
        self.assertAlmostEqual(from_crystal[0].weight, 0.7)
        self.assertEqual(from_crystal[0].context["origin"], "extraction")

        to_entity = self.db.relations_to("entity", 2)
        self.assertEqual(len(to_entity), 1)

    def test_all_relations(self):
        for i in range(3):
            self.db.insert_relation(Relation(source_type="crystal", source_id=i, target_type="entity", target_id=i, relation="mentions"))
        self.assertEqual(len(self.db.all_relations()), 3)


class TestSQLiteBackendContradictions(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")

    def tearDown(self):
        self.db.close()

    def test_insert_and_query(self):
        c = Contradiction(topic="language", claim_a="I like Python", claim_b="I hate Python", evidence_event_a=1, evidence_event_b=2)
        cid = self.db.insert_contradiction(c)
        self.assertIsInstance(cid, int)

        open_list = self.db.open_contradictions()
        self.assertEqual(len(open_list), 1)
        self.assertEqual(open_list[0].topic, "language")

    def test_update_resolution(self):
        cid = self.db.insert_contradiction(Contradiction(topic="test", claim_a="a", claim_b="b"))
        contras = self.db.open_contradictions()
        contras[0].resolution_state = "resolved_a"
        contras[0].resolution_ts = "2024-06-01T00:00:00"
        self.db.update_contradiction(contras[0])

        self.assertEqual(len(self.db.open_contradictions()), 0)
        all_c = self.db.all_contradictions()
        self.assertEqual(all_c[0].resolution_state, "resolved_a")

    def test_update_without_id_raises(self):
        c = Contradiction(topic="test", claim_a="a", claim_b="b")
        with self.assertRaises(ValueError):
            self.db.update_contradiction(c)


class TestSQLiteBackendActivations(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")

    def tearDown(self):
        self.db.close()

    def test_insert_and_recent(self):
        a = Activation(query="test query", active_self_state="builder", crystal_ids=[1, 2], entity_ids=[3], scene={"key": "val"}, quality_score=0.8)
        aid = self.db.insert_activation(a)
        self.assertIsInstance(aid, int)

        recent = self.db.recent_activations(limit=10)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].query, "test query")
        self.assertEqual(recent[0].crystal_ids, [1, 2])
        self.assertAlmostEqual(recent[0].quality_score, 0.8)


class TestSQLiteBackendProspective(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")

    def tearDown(self):
        self.db.close()

    def test_insert_and_active(self):
        pm = ProspectiveMemory(trigger_description="when deployment mentioned", payload_crystal_id=5, trigger_embedding=[0.1, 0.2])
        pid = self.db.insert_prospective(pm)
        self.assertIsInstance(pid, int)

        active = self.db.active_prospective_memories()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].trigger_description, "when deployment mentioned")
        self.assertEqual(active[0].trigger_embedding, [0.1, 0.2])
        self.assertFalse(active[0].fired)

    def test_fire(self):
        pid = self.db.insert_prospective(ProspectiveMemory(trigger_description="test", payload_crystal_id=1))
        self.db.fire_prospective(pid, "2024-06-01T12:00:00")
        active = self.db.active_prospective_memories()
        self.assertEqual(len(active), 0)


class TestSQLiteBackendSchemas(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")

    def tearDown(self):
        self.db.close()

    def test_insert_and_query(self):
        s = Schema(name="debugging", description="how user debugs", pattern="logs -> reproduce -> isolate", source_crystal_ids=[1, 2, 3], slot_definitions={"step1": "check logs"})
        sid = self.db.insert_schema(s)
        self.assertIsInstance(sid, int)

        all_s = self.db.all_schemas()
        self.assertEqual(len(all_s), 1)
        self.assertEqual(all_s[0].name, "debugging")
        self.assertEqual(all_s[0].source_crystal_ids, [1, 2, 3])
        self.assertEqual(all_s[0].slot_definitions["step1"], "check logs")

    def test_update(self):
        sid = self.db.insert_schema(Schema(name="test", description="v1"))
        schemas = self.db.all_schemas()
        schemas[0].description = "v2"
        schemas[0].activation_count = 10
        self.db.update_schema(schemas[0])
        refreshed = self.db.all_schemas()
        self.assertEqual(refreshed[0].description, "v2")
        self.assertEqual(refreshed[0].activation_count, 10)


class TestSQLiteBackendAntibodies(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")

    def tearDown(self):
        self.db.close()

    def test_insert_and_active(self):
        ab = AntibodyMemory(trigger_description="bad migration advice", suppress_crystal_id=7, reason="user corrected", trigger_embedding=[0.5, 0.6])
        abid = self.db.insert_antibody(ab)
        self.assertIsInstance(abid, int)

        active = self.db.active_antibodies()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].suppress_crystal_id, 7)
        self.assertEqual(active[0].reason, "user corrected")
        self.assertTrue(active[0].active)

    def test_inactive_not_returned(self):
        self.db.insert_antibody(AntibodyMemory(trigger_description="test", suppress_crystal_id=1, active=False))
        self.assertEqual(len(self.db.active_antibodies()), 0)


class TestSQLiteBackendExport(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteBackend(":memory:")

    def tearDown(self):
        self.db.close()

    def test_export_all_keys(self):
        # Insert one of each type
        self.db.set_meta("version", "test")
        self.db.insert_event(Event(actor="user", kind="chat", content="hello"))
        self.db.upsert_entity(Entity(name="python"))
        self.db.insert_crystal(Crystal(title="test", theme="test", summary="test"))
        self.db.insert_relation(Relation(source_type="crystal", source_id=1, target_type="entity", target_id=1, relation="mentions"))
        self.db.insert_contradiction(Contradiction(topic="test", claim_a="a", claim_b="b"))
        self.db.insert_activation(Activation(query="test"))
        self.db.insert_prospective(ProspectiveMemory(trigger_description="test", payload_crystal_id=1))
        self.db.insert_schema(Schema(name="test"))
        self.db.insert_antibody(AntibodyMemory(trigger_description="test", suppress_crystal_id=1))

        export = self.db.export_all()
        expected_keys = {"meta", "events", "entities", "crystals", "relations", "contradictions", "activations", "prospective_memories", "schemas", "antibodies"}
        self.assertEqual(set(export.keys()), expected_keys)
        self.assertEqual(len(export["events"]), 1)
        self.assertEqual(len(export["entities"]), 1)
        self.assertEqual(len(export["crystals"]), 1)


class TestSQLiteBackendContextManager(unittest.TestCase):
    def test_context_manager(self):
        with SQLiteBackend(":memory:") as db:
            db.set_meta("key", "value")
            self.assertEqual(db.get_meta("key"), "value")


if __name__ == "__main__":
    unittest.main()
