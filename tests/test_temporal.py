"""Tests for the temporal engine — decay, clustering, gravity, prospective memory, versioning."""

import unittest

from semantic_gravity_memory.models import (
    Crystal,
    ProspectiveMemory,
    Relation,
    SalienceVector,
)
from semantic_gravity_memory.storage.sqlite_backend import SQLiteBackend
from semantic_gravity_memory.core.temporal import (
    crystal_strength,
    reinforce_crystal,
    cluster_crystals,
    create_episode_relations,
    get_episode_members,
    temporal_gravity,
    temporal_proximity_bonus,
    recency_score,
    create_prospective,
    check_prospective_triggers,
    fire_prospective,
    fire_all_triggered,
    version_crystal,
    get_crystal_history,
    belief_at_version,
    decay_all_crystals,
    auto_cluster,
)


def _high_salience() -> SalienceVector:
    """Convenience: a crystal that scores well on combined()."""
    return SalienceVector(
        emotional=0.8, practical=0.8, identity=0.8,
        temporal=0.8, uncertainty=0.8, novelty=0.8,
    )


def _make_crystal(storage, title="test", ts="2024-06-01T12:00:00", **kw):
    defaults = dict(
        title=title, theme=title, summary=f"summary of {title}",
        created_ts=ts, salience=_high_salience(), confidence=0.8,
        decay_rate=0.1,
    )
    defaults.update(kw)
    cid = storage.insert_crystal(Crystal(**defaults))
    return storage.get_crystal(cid)


# =========================================================================
# 1. Crystal Strength
# =========================================================================


class TestCrystalStrength(unittest.TestCase):
    def test_fresh_crystal_has_high_strength(self):
        c = Crystal(
            created_ts="2024-06-01T12:00:00",
            salience=_high_salience(),
            confidence=0.8,
            decay_rate=0.1,
        )
        s = crystal_strength(c, now_ts="2024-06-01T12:00:00")
        self.assertGreater(s, 0.5)

    def test_strength_decays_over_time(self):
        c = Crystal(
            created_ts="2024-06-01T12:00:00",
            salience=_high_salience(),
            confidence=0.8,
            decay_rate=0.1,
        )
        s_fresh = crystal_strength(c, now_ts="2024-06-01T12:00:00")
        s_day = crystal_strength(c, now_ts="2024-06-02T12:00:00")
        s_week = crystal_strength(c, now_ts="2024-06-08T12:00:00")
        self.assertGreater(s_fresh, s_day)
        self.assertGreater(s_day, s_week)

    def test_zero_decay_rate_no_decay(self):
        c = Crystal(
            created_ts="2024-01-01T00:00:00",
            salience=_high_salience(),
            confidence=0.8,
            decay_rate=0.0,
        )
        s_old = crystal_strength(c, now_ts="2024-12-31T23:59:59")
        s_new = crystal_strength(c, now_ts="2024-01-01T00:00:00")
        self.assertAlmostEqual(s_old, s_new, places=4)

    def test_high_access_count_adds_reinforcement(self):
        c = Crystal(
            created_ts="2024-06-01T12:00:00",
            salience=_high_salience(),
            confidence=0.8,
            decay_rate=0.1,
            access_count=10,
            last_accessed_ts="2024-06-01T12:00:00",
        )
        s_accessed = crystal_strength(c, now_ts="2024-06-10T12:00:00")
        c_none = Crystal(
            created_ts="2024-06-01T12:00:00",
            salience=_high_salience(),
            confidence=0.8,
            decay_rate=0.1,
            access_count=0,
        )
        s_untouched = crystal_strength(c_none, now_ts="2024-06-10T12:00:00")
        self.assertGreater(s_accessed, s_untouched)

    def test_reinforcement_capped(self):
        c = Crystal(
            created_ts="2024-06-01T12:00:00",
            salience=_high_salience(),
            confidence=0.8,
            decay_rate=0.5,
            access_count=100,
            last_accessed_ts="2024-06-01T12:00:00",
        )
        s = crystal_strength(c, now_ts="2024-12-01T12:00:00")
        self.assertLessEqual(s, 1.0)

    def test_last_accessed_resets_decay_clock(self):
        # Crystal created long ago but accessed recently → still strong
        c = Crystal(
            created_ts="2020-01-01T00:00:00",
            salience=_high_salience(),
            confidence=0.8,
            decay_rate=0.1,
            access_count=5,
            last_accessed_ts="2024-06-01T11:55:00",
        )
        s = crystal_strength(c, now_ts="2024-06-01T12:00:00")
        self.assertGreater(s, 0.5)


# =========================================================================
# 2. Reinforcement
# =========================================================================


class TestReinforcement(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")

    def tearDown(self):
        self.storage.close()

    def test_bumps_access_count(self):
        c = _make_crystal(self.storage)
        self.assertEqual(c.access_count, 0)
        updated = reinforce_crystal(self.storage, c.id, now_ts="2024-06-02T00:00:00")
        self.assertEqual(updated.access_count, 1)

    def test_sets_last_accessed_ts(self):
        c = _make_crystal(self.storage)
        self.assertIsNone(c.last_accessed_ts)
        updated = reinforce_crystal(self.storage, c.id, now_ts="2024-06-02T00:00:00")
        self.assertEqual(updated.last_accessed_ts, "2024-06-02T00:00:00")

    def test_multiple_reinforcements_accumulate(self):
        c = _make_crystal(self.storage)
        reinforce_crystal(self.storage, c.id, now_ts="2024-06-02T01:00:00")
        reinforce_crystal(self.storage, c.id, now_ts="2024-06-02T02:00:00")
        reinforce_crystal(self.storage, c.id, now_ts="2024-06-02T03:00:00")
        refreshed = self.storage.get_crystal(c.id)
        self.assertEqual(refreshed.access_count, 3)
        self.assertEqual(refreshed.last_accessed_ts, "2024-06-02T03:00:00")

    def test_missing_crystal_raises(self):
        with self.assertRaises(ValueError):
            reinforce_crystal(self.storage, 9999)


# =========================================================================
# 3. Temporal Clustering
# =========================================================================


class TestTemporalClustering(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")

    def tearDown(self):
        self.storage.close()

    def test_close_crystals_cluster_together(self):
        crystals = [
            Crystal(id=1, created_ts="2024-06-01T10:00:00"),
            Crystal(id=2, created_ts="2024-06-01T11:00:00"),
            Crystal(id=3, created_ts="2024-06-01T13:00:00"),
        ]
        clusters = cluster_crystals(crystals, window_hours=4.0)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 3)

    def test_far_crystals_separate_clusters(self):
        crystals = [
            Crystal(id=1, created_ts="2024-06-01T08:00:00"),
            Crystal(id=2, created_ts="2024-06-01T09:00:00"),
            Crystal(id=3, created_ts="2024-06-01T20:00:00"),
            Crystal(id=4, created_ts="2024-06-01T21:00:00"),
        ]
        clusters = cluster_crystals(crystals, window_hours=4.0)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(clusters[0]), 2)
        self.assertEqual(len(clusters[1]), 2)

    def test_single_crystal_one_cluster(self):
        crystals = [Crystal(id=1, created_ts="2024-06-01T12:00:00")]
        clusters = cluster_crystals(crystals, window_hours=4.0)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 1)

    def test_empty_list(self):
        self.assertEqual(cluster_crystals([]), [])

    def test_unsorted_input_handled(self):
        crystals = [
            Crystal(id=3, created_ts="2024-06-01T15:00:00"),
            Crystal(id=1, created_ts="2024-06-01T08:00:00"),
            Crystal(id=2, created_ts="2024-06-01T09:00:00"),
        ]
        clusters = cluster_crystals(crystals, window_hours=4.0)
        # 08 and 09 cluster, 15 is separate
        self.assertEqual(len(clusters), 2)


class TestEpisodeRelations(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")

    def tearDown(self):
        self.storage.close()

    def test_creates_chain_relations(self):
        c1 = _make_crystal(self.storage, "a", ts="2024-06-01T08:00:00")
        c2 = _make_crystal(self.storage, "b", ts="2024-06-01T09:00:00")
        c3 = _make_crystal(self.storage, "c", ts="2024-06-01T10:00:00")
        clusters = [[c1, c2, c3]]
        count = create_episode_relations(self.storage, clusters)
        self.assertEqual(count, 2)  # c1→c2, c2→c3

    def test_no_duplicates_on_rerun(self):
        c1 = _make_crystal(self.storage, "a", ts="2024-06-01T08:00:00")
        c2 = _make_crystal(self.storage, "b", ts="2024-06-01T09:00:00")
        clusters = [[c1, c2]]
        create_episode_relations(self.storage, clusters)
        count2 = create_episode_relations(self.storage, clusters)
        self.assertEqual(count2, 0)

    def test_single_crystal_no_relations(self):
        c1 = _make_crystal(self.storage, "solo")
        count = create_episode_relations(self.storage, [[c1]])
        self.assertEqual(count, 0)

    def test_get_episode_members(self):
        c1 = _make_crystal(self.storage, "a", ts="2024-06-01T08:00:00")
        c2 = _make_crystal(self.storage, "b", ts="2024-06-01T09:00:00")
        c3 = _make_crystal(self.storage, "c", ts="2024-06-01T10:00:00")
        create_episode_relations(self.storage, [[c1, c2, c3]])
        members = get_episode_members(self.storage, c2.id)
        self.assertEqual(sorted(members), sorted([c1.id, c2.id, c3.id]))


# =========================================================================
# 4. Temporal Gravity
# =========================================================================


class TestTemporalGravity(unittest.TestCase):
    def test_same_time_max_gravity(self):
        g = temporal_gravity("2024-06-01T12:00:00", "2024-06-01T12:00:00")
        self.assertAlmostEqual(g, 1.0, places=4)

    def test_eight_hours_half(self):
        g = temporal_gravity(
            "2024-06-01T08:00:00", "2024-06-01T16:00:00",
            reference_hours=8.0,
        )
        self.assertAlmostEqual(g, 0.5, places=4)

    def test_far_apart_low_gravity(self):
        g = temporal_gravity(
            "2024-01-01T00:00:00", "2024-06-01T00:00:00",
            reference_hours=8.0,
        )
        self.assertLess(g, 0.01)

    def test_empty_ts_zero(self):
        self.assertAlmostEqual(temporal_gravity("", "2024-06-01T12:00:00"), 0.0)

    def test_proximity_bonus_on_crystal(self):
        c = Crystal(created_ts="2024-06-01T11:00:00")
        bonus = temporal_proximity_bonus(c, "2024-06-01T12:00:00", reference_hours=8.0)
        # 1 hour apart: 1 / (1 + 1/8) = 1/1.125 ≈ 0.889
        self.assertAlmostEqual(bonus, 1.0 / 1.125, places=3)


# =========================================================================
# 5. Recency Scoring
# =========================================================================


class TestRecencyScore(unittest.TestCase):
    def test_recent_high_score(self):
        s = recency_score("2024-06-01T12:00:00", now_ts="2024-06-01T12:00:00")
        self.assertAlmostEqual(s, 1.0, places=4)

    def test_half_life(self):
        s = recency_score(
            "2024-06-01T12:00:00", now_ts="2024-06-03T12:00:00",
            half_life_hours=48.0,
        )
        self.assertAlmostEqual(s, 0.5, places=4)

    def test_old_still_nonzero(self):
        s = recency_score(
            "2020-01-01T00:00:00", now_ts="2024-06-01T00:00:00",
        )
        self.assertGreater(s, 0.0)

    def test_empty_ts_zero(self):
        self.assertAlmostEqual(recency_score(""), 0.0)

    def test_recent_beats_old(self):
        recent = recency_score("2024-06-01T11:00:00", now_ts="2024-06-01T12:00:00")
        old = recency_score("2024-01-01T00:00:00", now_ts="2024-06-01T12:00:00")
        self.assertGreater(recent, old)


# =========================================================================
# 6. Prospective Memory
# =========================================================================


class TestProspectiveMemory(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")

    def tearDown(self):
        self.storage.close()

    def test_create_and_retrieve(self):
        pid = create_prospective(
            self.storage, "deployment", payload_crystal_id=42,
            trigger_embedding=[0.1, 0.9, 0.0],
        )
        self.assertIsInstance(pid, int)
        active = self.storage.active_prospective_memories()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].trigger_description, "deployment")

    def test_text_trigger_match(self):
        create_prospective(self.storage, "deployment", payload_crystal_id=1)
        triggered = check_prospective_triggers(
            self.storage, "We should talk about the deployment plan",
        )
        self.assertEqual(len(triggered), 1)

    def test_text_trigger_no_match(self):
        create_prospective(self.storage, "deployment", payload_crystal_id=1)
        triggered = check_prospective_triggers(
            self.storage, "The weather is nice",
        )
        self.assertEqual(len(triggered), 0)

    def test_embedding_trigger_match(self):
        create_prospective(
            self.storage, "related topic", payload_crystal_id=1,
            trigger_embedding=[1.0, 0.0, 0.0],
        )
        triggered = check_prospective_triggers(
            self.storage, "anything",
            embedding=[0.95, 0.05, 0.0],  # very similar
            similarity_threshold=0.9,
        )
        self.assertEqual(len(triggered), 1)

    def test_embedding_trigger_no_match(self):
        create_prospective(
            self.storage, "something specific", payload_crystal_id=1,
            trigger_embedding=[1.0, 0.0, 0.0],
        )
        triggered = check_prospective_triggers(
            self.storage, "unrelated",
            embedding=[0.0, 0.0, 1.0],  # orthogonal
            similarity_threshold=0.9,
        )
        # Text also doesn't match
        self.assertEqual(len(triggered), 0)

    def test_expired_skipped(self):
        create_prospective(
            self.storage, "deployment", payload_crystal_id=1,
            expiry_ts="2024-01-01T00:00:00",
        )
        triggered = check_prospective_triggers(
            self.storage, "deployment",
            now_ts="2024-06-01T00:00:00",
        )
        self.assertEqual(len(triggered), 0)

    def test_fire_removes_from_active(self):
        pid = create_prospective(self.storage, "deploy", payload_crystal_id=1)
        active = self.storage.active_prospective_memories()
        self.assertEqual(len(active), 1)
        fire_prospective(self.storage, active[0], now_ts="2024-06-01T12:00:00")
        self.assertEqual(len(self.storage.active_prospective_memories()), 0)

    def test_fire_all_returns_payload_ids(self):
        create_prospective(self.storage, "deploy", payload_crystal_id=10)
        create_prospective(self.storage, "release", payload_crystal_id=20)
        triggered = check_prospective_triggers(
            self.storage, "deploy the release now",
        )
        ids = fire_all_triggered(self.storage, triggered)
        self.assertIn(10, ids)
        self.assertIn(20, ids)
        self.assertEqual(len(self.storage.active_prospective_memories()), 0)


# =========================================================================
# 7. Memory Versioning
# =========================================================================


class TestMemoryVersioning(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")

    def tearDown(self):
        self.storage.close()

    def test_version_increments(self):
        c = _make_crystal(self.storage, "original")
        self.assertEqual(c.version, 1)
        new_v = version_crystal(
            self.storage, c.id,
            {"summary": "updated summary"},
            now_ts="2024-06-02T00:00:00",
        )
        self.assertEqual(new_v, 2)
        refreshed = self.storage.get_crystal(c.id)
        self.assertEqual(refreshed.version, 2)
        self.assertEqual(refreshed.summary, "updated summary")

    def test_history_preserved(self):
        c = _make_crystal(self.storage, "v1 title")
        version_crystal(self.storage, c.id, {"title": "v2 title"})
        version_crystal(self.storage, c.id, {"title": "v3 title"})
        history = get_crystal_history(self.storage, c.id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["title"], "v1 title")
        self.assertEqual(history[1]["title"], "v2 title")

    def test_belief_at_version(self):
        c = _make_crystal(self.storage, "alpha")
        version_crystal(self.storage, c.id, {"title": "beta"})
        snap = belief_at_version(self.storage, c.id, version=1)
        self.assertIsNotNone(snap)
        self.assertEqual(snap["title"], "alpha")

    def test_belief_at_missing_version(self):
        c = _make_crystal(self.storage, "only")
        snap = belief_at_version(self.storage, c.id, version=99)
        self.assertIsNone(snap)

    def test_version_preserves_salience(self):
        c = _make_crystal(self.storage, "sal test")
        version_crystal(self.storage, c.id, {"confidence": 0.95})
        history = get_crystal_history(self.storage, c.id)
        self.assertEqual(len(history), 1)
        sal = history[0]["salience"]
        self.assertAlmostEqual(sal["emotional"], 0.8)

    def test_missing_crystal_raises(self):
        with self.assertRaises(ValueError):
            version_crystal(self.storage, 9999, {"title": "x"})


# =========================================================================
# 8. Batch Helpers
# =========================================================================


class TestBatchHelpers(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteBackend(":memory:")

    def tearDown(self):
        self.storage.close()

    def test_decay_marks_weak_crystals_dormant(self):
        # Create a crystal with very high decay, then check far in the future
        c = _make_crystal(
            self.storage, "ephemeral",
            ts="2020-01-01T00:00:00",
            decay_rate=1.0,
            confidence=0.3,
            salience=SalienceVector(practical=0.1),
        )
        checked, dormant = decay_all_crystals(
            self.storage, now_ts="2024-06-01T00:00:00",
        )
        self.assertEqual(checked, 1)
        self.assertEqual(dormant, 1)
        refreshed = self.storage.get_crystal(c.id)
        self.assertIsNotNone(refreshed.valid_to_ts)

    def test_decay_keeps_strong_crystals(self):
        _make_crystal(
            self.storage, "strong",
            ts="2024-06-01T11:00:00",
            decay_rate=0.01,
        )
        checked, dormant = decay_all_crystals(
            self.storage, now_ts="2024-06-01T12:00:00",
        )
        self.assertEqual(checked, 1)
        self.assertEqual(dormant, 0)

    def test_auto_cluster(self):
        _make_crystal(self.storage, "a", ts="2024-06-01T08:00:00")
        _make_crystal(self.storage, "b", ts="2024-06-01T09:00:00")
        _make_crystal(self.storage, "c", ts="2024-06-01T20:00:00")
        count = auto_cluster(self.storage, window_hours=4.0)
        # a-b cluster → 1 relation.  c alone → 0
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
