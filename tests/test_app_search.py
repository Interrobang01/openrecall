import os
import tempfile
import unittest
from unittest import mock

from openrecall.app import (
    _entry_matches_exact_phrases,
    _parse_embedding_expression,
    _parse_search_query,
    _resolve_search_metric,
    app,
)
import openrecall.screenshot as screenshot


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


class TestEmbeddingExpressionParsing(unittest.TestCase):
    def test_parse_embedding_expression_add_subtract(self):
        parsed = _parse_embedding_expression("(queen) - (king) + (woman)")
        self.assertEqual(parsed, [("queen", 1), ("king", -1), ("woman", 1)])

    def test_parse_embedding_expression_requires_parentheses(self):
        parsed = _parse_embedding_expression("queen - king")
        self.assertIsNone(parsed)

    def test_parse_embedding_expression_ignores_quoted_parentheses(self):
        parsed = _parse_embedding_expression('"(queen) - (king)"')
        self.assertIsNone(parsed)

    def test_parse_embedding_expression_rejects_malformed_expression(self):
        parsed = _parse_embedding_expression("(queen) - king")
        self.assertIsNone(parsed)


class TestSearchMetricResolution(unittest.TestCase):
    def test_resolve_metric_accepts_supported_values(self):
        self.assertEqual(_resolve_search_metric("dot"), "dot")
        self.assertEqual(_resolve_search_metric("euclidean"), "euclidean")

    def test_resolve_metric_falls_back_to_cosine(self):
        self.assertEqual(_resolve_search_metric("unsupported"), "cosine")
        self.assertEqual(_resolve_search_metric(""), "cosine")


class TestCapturePauseApi(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()
        screenshot.capture_state["paused_until_ts"] = 0
        screenshot.capture_state["paused_indefinitely"] = False

    def test_pause_forever_then_resume(self):
        with mock.patch.object(screenshot, "send_system_notification", return_value=True):
            pause_response = self.client.post("/api/capture/pause-forever", json={})
            self.assertEqual(pause_response.status_code, 200)
            self.assertTrue(pause_response.get_json().get("paused_indefinitely"))

            status_response = self.client.get("/api/status")
            self.assertEqual(status_response.status_code, 200)
            status_payload = status_response.get_json()
            self.assertTrue(status_payload.get("is_paused"))
            self.assertTrue(status_payload.get("paused_indefinitely"))

            resume_response = self.client.post("/api/capture/resume", json={})
            self.assertEqual(resume_response.status_code, 200)

            status_after_resume = self.client.get("/api/status").get_json()
            self.assertFalse(status_after_resume.get("is_paused"))
            self.assertFalse(status_after_resume.get("paused_indefinitely"))


class TestFrameEndpointPendingFallback(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_serves_pending_frame_when_segment_not_finalized(self):
        with tempfile.TemporaryDirectory() as tmp_segments, tempfile.TemporaryDirectory() as tmp_pending, mock.patch(
            "openrecall.app.segments_path",
            tmp_segments,
        ), mock.patch(
            "openrecall.app.pending_frames_path",
            tmp_pending,
        ):
            pending_filename = "123_m1.webp"
            with open(os.path.join(tmp_pending, pending_filename), "wb") as pending_file:
                pending_file.write(b"RIFFdummywebp")

            response = self.client.get(
                "/frame?segment=123_m1.mkv&pts_ms=0&thumb=123_m1.webp"
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.data, b"RIFFdummywebp")

    def test_returns_not_found_when_segment_and_pending_missing(self):
        with tempfile.TemporaryDirectory() as tmp_segments, tempfile.TemporaryDirectory() as tmp_pending, mock.patch(
            "openrecall.app.segments_path",
            tmp_segments,
        ), mock.patch(
            "openrecall.app.pending_frames_path",
            tmp_pending,
        ):
            response = self.client.get(
                "/frame?segment=999_m1.mkv&pts_ms=0&thumb=999_m1.webp"
            )

            self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
