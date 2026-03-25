"""Tests for contradiction detection."""

import unittest

from semantic_gravity_memory.core.contradiction import ContradictionDetector
from semantic_gravity_memory.models import Contradiction, Event
from semantic_gravity_memory.storage.sqlite_backend import SQLiteBackend


class TestPreferenceContradictions(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.detector = ContradictionDetector(self.storage)

    def tearDown(self):
        self.storage.close()

    def test_like_vs_hate(self):
        self.storage.insert_event(Event(
            actor="user", kind="chat",
            content="I like JavaScript for frontend work",
        ))
        eid2 = self.storage.insert_event(Event(
            actor="user", kind="chat",
            content="I hate JavaScript, it's terrible",
        ))
        contras = self.detector.check_preferences(
            "I hate JavaScript, it's terrible", eid2,
        )
        self.assertGreater(len(contras), 0)
        self.assertEqual(contras[0].resolution_state, "open")

    def test_no_contradiction_different_subjects(self):
        self.storage.insert_event(Event(
            actor="user", kind="chat", content="I like Python",
        ))
        eid2 = self.storage.insert_event(Event(
            actor="user", kind="chat", content="I hate JavaScript",
        ))
        contras = self.detector.check_preferences("I hate JavaScript", eid2)
        self.assertEqual(len(contras), 0)

    def test_growth_not_flagged(self):
        self.storage.insert_event(Event(
            actor="user", kind="chat", content="I like simple code",
        ))
        text = "I used to like simple code but now I appreciate complexity"
        eid2 = self.storage.insert_event(Event(
            actor="user", kind="chat", content=text,
        ))
        contras = self.detector.check_preferences(text, eid2)
        self.assertEqual(len(contras), 0)

    def test_prefer_vs_avoid(self):
        self.storage.insert_event(Event(
            actor="user", kind="chat", content="I prefer tabs for indentation",
        ))
        eid2 = self.storage.insert_event(Event(
            actor="user", kind="chat", content="I avoid tabs, spaces only",
        ))
        contras = self.detector.check_preferences(
            "I avoid tabs, spaces only", eid2,
        )
        self.assertGreater(len(contras), 0)


class TestFactualContradictions(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.detector = ContradictionDetector(self.storage)

    def tearDown(self):
        self.storage.close()

    def test_conflicting_tool_claims(self):
        self.storage.insert_event(Event(
            actor="user", kind="chat", content="our API uses REST",
        ))
        eid2 = self.storage.insert_event(Event(
            actor="user", kind="chat", content="our API uses GraphQL",
        ))
        contras = self.detector.check_factual("our API uses GraphQL", eid2)
        self.assertGreater(len(contras), 0)

    def test_same_claim_no_contradiction(self):
        self.storage.insert_event(Event(
            actor="user", kind="chat", content="the app uses Python",
        ))
        eid2 = self.storage.insert_event(Event(
            actor="user", kind="chat", content="the app uses Python",
        ))
        contras = self.detector.check_factual("the app uses Python", eid2)
        self.assertEqual(len(contras), 0)


class TestTemporalContradictions(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.detector = ContradictionDetector(self.storage)

    def tearDown(self):
        self.storage.close()

    def test_missed_commitment(self):
        self.storage.insert_event(Event(
            actor="user", kind="chat",
            content="I'll finish the report by Friday",
        ))
        text = "I still haven't finished the report"
        eid2 = self.storage.insert_event(Event(
            actor="user", kind="chat", content=text,
        ))
        contras = self.detector.check_temporal(text, eid2)
        self.assertGreater(len(contras), 0)
        self.assertIn("temporal", contras[0].notes)

    def test_no_incompletion_signal(self):
        self.storage.insert_event(Event(
            actor="user", kind="chat",
            content="I'll finish the report by Friday",
        ))
        # Mentioning the report without incompletion shouldn't trigger
        text = "The report is looking good"
        eid2 = self.storage.insert_event(Event(
            actor="user", kind="chat", content=text,
        ))
        contras = self.detector.check_temporal(text, eid2)
        self.assertEqual(len(contras), 0)


class TestCheckAll(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.detector = ContradictionDetector(self.storage)

    def tearDown(self):
        self.storage.close()

    def test_combines_and_deduplicates(self):
        self.storage.insert_event(Event(
            actor="user", kind="chat", content="I like Python",
        ))
        eid2 = self.storage.insert_event(Event(
            actor="user", kind="chat", content="I hate Python",
        ))
        contras = self.detector.check_all("I hate Python", eid2)
        self.assertGreater(len(contras), 0)
        # No duplicates
        keys = [(c.topic, c.evidence_event_a, c.evidence_event_b) for c in contras]
        self.assertEqual(len(set(keys)), len(keys))

    def test_empty_when_no_history(self):
        eid = self.storage.insert_event(Event(
            actor="user", kind="chat", content="I like Python",
        ))
        contras = self.detector.check_all("I like Python", eid)
        self.assertEqual(len(contras), 0)


class TestResolutionSuggestion(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")
        self.detector = ContradictionDetector(self.storage)

    def tearDown(self):
        self.storage.close()

    def test_suggests_newer_wins(self):
        eid1 = self.storage.insert_event(Event(
            ts="2024-01-01T00:00:00", actor="user", kind="chat", content="old",
        ))
        eid2 = self.storage.insert_event(Event(
            ts="2024-06-01T00:00:00", actor="user", kind="chat", content="new",
        ))
        contra = Contradiction(
            topic="test", claim_a="old", claim_b="new",
            evidence_event_a=eid1, evidence_event_b=eid2,
        )
        suggestion = self.detector.suggest_resolution(contra)
        self.assertIsNotNone(suggestion)
        self.assertIn("resolved_b", suggestion)

    def test_none_for_missing_events(self):
        contra = Contradiction(
            topic="test", claim_a="a", claim_b="b",
            evidence_event_a=999, evidence_event_b=998,
        )
        suggestion = self.detector.suggest_resolution(contra)
        self.assertIsNone(suggestion)

    def test_none_for_no_evidence(self):
        contra = Contradiction(topic="test", claim_a="a", claim_b="b")
        suggestion = self.detector.suggest_resolution(contra)
        self.assertIsNone(suggestion)


if __name__ == "__main__":
    unittest.main()
