"""Tests for entity extraction."""

import unittest

from semantic_gravity_memory.core.entity_extractor import (
    extract_entities,
    extract_relationships,
    find_co_occurrences,
)


class TestCapitalizedPhrases(unittest.TestCase):
    def test_multi_word_proper_noun(self):
        entities = extract_entities("I've been using Google Cloud Platform for deployments")
        names = [e[0] for e in entities]
        self.assertTrue(any("Google Cloud Platform" in n for n in names))

    def test_two_word_name(self):
        entities = extract_entities("We partnered with South Bay last month")
        names_lower = [e[0].lower() for e in entities]
        self.assertTrue(any("south bay" in n for n in names_lower))


class TestCamelCase(unittest.TestCase):
    def test_pascal_case(self):
        entities = extract_entities("The QuickBooks integration is broken")
        names = [e[0] for e in entities]
        self.assertTrue(any("QuickBooks" in n for n in names))

    def test_allcaps_acronym(self):
        entities = extract_entities("We need to configure the AWS bucket")
        names = [e[0] for e in entities]
        self.assertTrue(any("AWS" in n for n in names))


class TestTechNames(unittest.TestCase):
    def test_single_tech(self):
        entities = extract_entities("I prefer Python for building prototypes")
        names = [e[0].lower() for e in entities]
        self.assertIn("python", names)

    def test_multiple_tools(self):
        entities = extract_entities("Deploy the flask app to docker with postgres")
        names = [e[0].lower() for e in entities]
        self.assertIn("flask", names)
        self.assertIn("docker", names)
        self.assertIn("postgres", names)

    def test_classified_as_tool(self):
        entities = extract_entities("I use sqlite for everything")
        kinds = {e[0].lower(): e[1] for e in entities}
        self.assertEqual(kinds.get("sqlite"), "tool")


class TestQuotedStrings(unittest.TestCase):
    def test_double_quotes(self):
        entities = extract_entities('The project is called "Moonshot Alpha"')
        names = [e[0] for e in entities]
        self.assertTrue(any("Moonshot Alpha" in n for n in names))


class TestDeduplication(unittest.TestCase):
    def test_case_variants_merge(self):
        entities = extract_entities("Python is great. I love python. PYTHON forever.")
        python_entries = [e for e in entities if "python" in e[0].lower()]
        self.assertEqual(len(python_entries), 1)

    def test_version_suffix_merge(self):
        entities = extract_entities("I use python3 daily, Python is my main language")
        python_entries = [e for e in entities if "python" in e[0].lower()]
        self.assertEqual(len(python_entries), 1)

    def test_substring_merge(self):
        # "postgres" and "postgresql" should merge
        entities = extract_entities("We use postgres. Actually postgresql is the full name.")
        pg_entries = [e for e in entities if "postgres" in e[0].lower()]
        self.assertEqual(len(pg_entries), 1)


class TestRelationshipExtraction(unittest.TestCase):
    def test_deployed_to(self):
        rels = extract_relationships("We deployed the app to Heroku last night")
        self.assertTrue(any(r[1] == "deployed_to" for r in rels))

    def test_built_with(self):
        rels = extract_relationships("I built the dashboard with React")
        self.assertTrue(any(r[1] == "built_with" for r in rels))

    def test_uses(self):
        rels = extract_relationships("The backend uses Redis for caching")
        self.assertTrue(any(r[1] == "uses" for r in rels))

    def test_no_relationships_in_plain_text(self):
        rels = extract_relationships("The weather is nice today")
        self.assertEqual(len(rels), 0)


class TestCoOccurrence(unittest.TestCase):
    def test_pair_count(self):
        entities = [("flask", "tool"), ("postgres", "tool"), ("docker", "tool")]
        pairs = find_co_occurrences(entities)
        self.assertEqual(len(pairs), 3)  # C(3,2) = 3

    def test_single_entity_no_pairs(self):
        entities = [("flask", "tool")]
        pairs = find_co_occurrences(entities)
        self.assertEqual(len(pairs), 0)

    def test_pairs_are_sorted(self):
        entities = [("zebra", "concept"), ("alpha", "concept")]
        pairs = find_co_occurrences(entities)
        self.assertEqual(pairs[0], ("alpha", "zebra"))


class TestEdgeCases(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(extract_entities(""), [])

    def test_only_stopwords(self):
        entities = extract_entities("this is a very simple and basic thing to do")
        self.assertLessEqual(len(entities), 3)

    def test_long_repeated_text(self):
        text = "Python and Flask " * 100
        entities = extract_entities(text)
        self.assertLessEqual(len(entities), 20)

    def test_urls_classified(self):
        entities = extract_entities('Check https://example.com for details')
        url_entities = [e for e in entities if e[1] == "url"]
        # URL might or might not be extracted depending on tokenization
        # but if it is, it should be classified as url
        for e in entities:
            if "http" in e[0]:
                self.assertEqual(e[1], "url")


if __name__ == "__main__":
    unittest.main()
