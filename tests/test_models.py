"""Tests for data models and SalienceVector."""

import unittest
from semantic_gravity_memory.models import (
    SalienceVector,
    Event,
    Entity,
    Crystal,
    Relation,
    Contradiction,
    Activation,
    ProspectiveMemory,
    Schema,
    AntibodyMemory,
)


class TestSalienceVector(unittest.TestCase):
    def test_defaults_are_zero(self):
        sv = SalienceVector()
        self.assertEqual(sv.emotional, 0.0)
        self.assertEqual(sv.novelty, 0.0)

    def test_combined_default_weights(self):
        sv = SalienceVector(emotional=1.0, practical=1.0, identity=1.0, temporal=1.0, uncertainty=1.0, novelty=1.0)
        # weights sum to 1.0, so combined of all-ones should be 1.0
        self.assertAlmostEqual(sv.combined(), 1.0, places=5)

    def test_combined_custom_weights(self):
        sv = SalienceVector(practical=0.8)
        score = sv.combined({"emotional": 0, "practical": 1.0, "identity": 0, "temporal": 0, "uncertainty": 0, "novelty": 0})
        self.assertAlmostEqual(score, 0.8, places=5)

    def test_combined_zero_vector(self):
        sv = SalienceVector()
        self.assertAlmostEqual(sv.combined(), 0.0, places=5)

    def test_peak_dimension(self):
        sv = SalienceVector(emotional=0.1, practical=0.9, temporal=0.3)
        self.assertEqual(sv.peak_dimension(), "practical")

    def test_peak_dimension_tie(self):
        sv = SalienceVector(emotional=0.5, practical=0.5)
        # any of the tied dimensions is acceptable
        self.assertIn(sv.peak_dimension(), ["emotional", "practical"])

    def test_to_dict_round_trip(self):
        sv = SalienceVector(emotional=0.3, practical=0.7, novelty=0.1)
        d = sv.to_dict()
        sv2 = SalienceVector.from_dict(d)
        self.assertAlmostEqual(sv.emotional, sv2.emotional)
        self.assertAlmostEqual(sv.practical, sv2.practical)
        self.assertAlmostEqual(sv.novelty, sv2.novelty)

    def test_from_dict_none(self):
        sv = SalienceVector.from_dict(None)
        self.assertEqual(sv.emotional, 0.0)

    def test_from_dict_partial(self):
        sv = SalienceVector.from_dict({"practical": 0.5})
        self.assertEqual(sv.practical, 0.5)
        self.assertEqual(sv.emotional, 0.0)


class TestEventModel(unittest.TestCase):
    def test_defaults(self):
        e = Event()
        self.assertIsNone(e.id)
        self.assertEqual(e.actor, "")
        self.assertEqual(e.context, {})
        self.assertIsNone(e.embedding)

    def test_fields(self):
        e = Event(ts="2024-01-01T00:00:00", actor="user", kind="chat", content="hello", salience=0.5)
        self.assertEqual(e.actor, "user")
        self.assertEqual(e.salience, 0.5)


class TestCrystalModel(unittest.TestCase):
    def test_defaults(self):
        c = Crystal()
        self.assertIsNone(c.id)
        self.assertEqual(c.memory_type, "episodic")
        self.assertEqual(c.access_count, 0)
        self.assertAlmostEqual(c.decay_rate, 0.1)
        self.assertEqual(c.version, 1)
        self.assertIsNone(c.parent_crystal_id)
        self.assertIsInstance(c.salience, SalienceVector)

    def test_salience_is_independent(self):
        c1 = Crystal()
        c2 = Crystal()
        c1.salience.practical = 0.9
        # c2 should have its own SalienceVector
        self.assertAlmostEqual(c2.salience.practical, 0.0)


class TestEntityModel(unittest.TestCase):
    def test_defaults(self):
        e = Entity(name="python")
        self.assertEqual(e.kind, "concept")
        self.assertEqual(e.mention_count, 0)


class TestRelationModel(unittest.TestCase):
    def test_fields(self):
        r = Relation(source_type="crystal", source_id=1, target_type="entity", target_id=2, relation="mentions", weight=0.8)
        self.assertEqual(r.relation, "mentions")
        self.assertAlmostEqual(r.weight, 0.8)


class TestContradictionModel(unittest.TestCase):
    def test_defaults(self):
        c = Contradiction(topic="language", claim_a="I like Python", claim_b="I hate Python")
        self.assertEqual(c.resolution_state, "open")
        self.assertIsNone(c.resolution_ts)


class TestProspectiveMemoryModel(unittest.TestCase):
    def test_defaults(self):
        pm = ProspectiveMemory(trigger_description="deployment", payload_crystal_id=5)
        self.assertFalse(pm.fired)
        self.assertIsNone(pm.fired_ts)


class TestSchemaModel(unittest.TestCase):
    def test_defaults(self):
        s = Schema(name="debugging_pattern")
        self.assertEqual(s.activation_count, 0)
        self.assertEqual(s.source_crystal_ids, [])


class TestAntibodyModel(unittest.TestCase):
    def test_defaults(self):
        a = AntibodyMemory(trigger_description="bad advice", suppress_crystal_id=3, reason="user corrected")
        self.assertTrue(a.active)


if __name__ == "__main__":
    unittest.main()
