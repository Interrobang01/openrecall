import os
import tempfile
import unittest
from unittest import mock

from openrecall.app import (
    _apply_proximity_dedup,
    _entry_matches_exact_phrases,
    _entry_matches_monitor_filter,
    _entry_matches_window_filter,
    _entry_in_date_range,
    _format_proximity_human,
    _parse_datetime_local_to_timestamp,
    _parse_embedding_expression,
    _proximity_level_to_seconds,
    _proximity_seconds_to_level,
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


class TestSearchFilters(unittest.TestCase):
    def test_parse_datetime_local_to_timestamp(self):
        parsed = _parse_datetime_local_to_timestamp("2026-03-23T15:40")
        self.assertIsInstance(parsed, int)
        self.assertGreater(parsed, 0)

    def test_parse_datetime_local_to_timestamp_invalid(self):
        self.assertIsNone(_parse_datetime_local_to_timestamp("bad-date"))
        self.assertIsNone(_parse_datetime_local_to_timestamp(""))

    def test_entry_in_date_range(self):
        self.assertTrue(_entry_in_date_range(100, None, None))
        self.assertTrue(_entry_in_date_range(100, 50, 100))
        self.assertFalse(_entry_in_date_range(49, 50, 100))
        self.assertFalse(_entry_in_date_range(101, 50, 100))

    def test_window_filter_matches_app_or_title(self):
        self.assertTrue(_entry_matches_window_filter("Firefox", "OpenRecall - Search", "recall"))
        self.assertTrue(_entry_matches_window_filter("Code", "README.md", "readme"))
        self.assertFalse(_entry_matches_window_filter("Code", "README.md", "invoice"))

    def test_monitor_filter_matches_expected_monitor(self):
        self.assertTrue(_entry_matches_monitor_filter(2, None))
        self.assertTrue(_entry_matches_monitor_filter(2, 2))
        self.assertFalse(_entry_matches_monitor_filter(2, 1))

    def test_proximity_dedup_keeps_top_ranked_first(self):
        ranked_results = [
            {"timestamp": 1000, "score": "A"},
            {"timestamp": 1003, "score": "B"},
            {"timestamp": 1008, "score": "C"},
            {"timestamp": 1012, "score": "D"},
        ]
        deduped = _apply_proximity_dedup(ranked_results, proximity_seconds=5)
        self.assertEqual([item["score"] for item in deduped], ["A", "C"])

    def test_proximity_human_formatting(self):
        self.assertEqual(_format_proximity_human(0), "Off")
        self.assertEqual(_format_proximity_human(45), "45s")
        self.assertEqual(_format_proximity_human(120), "2m")
        self.assertEqual(_format_proximity_human(3600), "1h")

    def test_logarithmic_proximity_level_round_trip(self):
        max_seconds = 31536000
        for seconds in (1, 15, 120, 3600, 86400, 31536000):
            level = _proximity_seconds_to_level(seconds, max_seconds)
            round_trip = _proximity_level_to_seconds(level, max_seconds)
            self.assertGreaterEqual(round_trip, 1)
            self.assertLessEqual(round_trip, max_seconds)

    def test_logarithmic_proximity_level_endpoints(self):
        max_seconds = 3600
        self.assertEqual(_proximity_level_to_seconds(0, max_seconds), 1)
        self.assertEqual(_proximity_level_to_seconds(1000, max_seconds), max_seconds)


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
            self.assertEqual(response.headers.get("X-OpenRecall-Frame-Source"), "pending_webp")

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


class TestOpenMediaFileApi(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_open_media_file_success_for_thumbnail(self):
        with tempfile.TemporaryDirectory() as tmp_thumbs, tempfile.TemporaryDirectory() as tmp_pending, tempfile.TemporaryDirectory() as tmp_segments, mock.patch(
            "openrecall.app.thumbnails_path",
            tmp_thumbs,
        ), mock.patch(
            "openrecall.app.pending_frames_path",
            tmp_pending,
        ), mock.patch(
            "openrecall.app.segments_path",
            tmp_segments,
        ), mock.patch("openrecall.app.subprocess.Popen") as popen_mock:
            thumb_name = "200_m1.webp"
            with open(os.path.join(tmp_thumbs, thumb_name), "wb") as thumb_file:
                thumb_file.write(b"thumb")

            response = self.client.post(
                "/api/open-media-file",
                json={
                    "thumb_filename": thumb_name,
                    "source": "thumbnail",
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.get_json().get("ok"))
            popen_mock.assert_called_once()

    def test_open_media_file_returns_not_found_for_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp_thumbs, tempfile.TemporaryDirectory() as tmp_pending, tempfile.TemporaryDirectory() as tmp_segments, mock.patch(
            "openrecall.app.thumbnails_path",
            tmp_thumbs,
        ), mock.patch(
            "openrecall.app.pending_frames_path",
            tmp_pending,
        ), mock.patch(
            "openrecall.app.segments_path",
            tmp_segments,
        ):
            response = self.client.post(
                "/api/open-media-file",
                json={
                    "segment_filename": "999_m1.mkv",
                    "thumb_filename": "999_m1.webp",
                    "source": "video_frame",
                },
            )

            self.assertEqual(response.status_code, 404)
            self.assertFalse(response.get_json().get("ok"))


if __name__ == "__main__":
    unittest.main()
