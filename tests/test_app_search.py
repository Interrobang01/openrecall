import unittest

from openrecall.app import _entry_matches_exact_phrases, _parse_search_query


class TestSearchQueryParsing(unittest.TestCase):
    def test_parse_plain_query(self):
        semantic_query, exact_phrases = _parse_search_query("invoice due date")
        self.assertEqual(semantic_query, "invoice due date")
        self.assertEqual(exact_phrases, [])

    def test_parse_exact_phrase(self):
        semantic_query, exact_phrases = _parse_search_query('report "error 500"')
        self.assertEqual(semantic_query, "report error 500")
        self.assertEqual(exact_phrases, ["error 500"])

    def test_parse_escaped_quotes(self):
        semantic_query, exact_phrases = _parse_search_query('literal \\"quote\\" test')
        self.assertEqual(semantic_query, 'literal "quote" test')
        self.assertEqual(exact_phrases, [])

    def test_exact_phrase_match_is_case_insensitive(self):
        text = "User saw Error 500 on dashboard"
        self.assertTrue(_entry_matches_exact_phrases(text, ["error 500"]))
        self.assertFalse(_entry_matches_exact_phrases(text, ["error 404"]))


if __name__ == "__main__":
    unittest.main()
