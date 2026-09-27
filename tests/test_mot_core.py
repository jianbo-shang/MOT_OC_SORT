import unittest
from unittest.mock import patch

import numpy as np

from mot_core import (Detection, FrameAnalyzer, LineCounter, MultiObjectTracker,
                      OcSortTrack, OcSortTracker, iou_matrix)


def detection(x1, y1, x2, y2, class_id=2, score=0.9):
    return Detection(np.array([x1, y1, x2, y2], dtype=np.float32), score, class_id)


class FakeAppearanceEncoder:
    def __init__(self):
        self.calls = 0
        self.crops = 0
        self.elapsed_ms = 0.0

    def encode(self, _frame, boxes):
        self.calls += 1
        self.crops += len(boxes)
        features = []
        for box in boxes:
            center_x = (float(box[0]) + float(box[2])) * 0.5
            features.append([1.0, 0.0] if center_x < 25 else [0.0, 1.0])
        return np.asarray(features, dtype=np.float32)


class TrackerTests(unittest.TestCase):
    def test_iou_matrix(self):
        result = iou_matrix(np.array([[0, 0, 10, 10]]), np.array([[5, 5, 15, 15]]))
        self.assertAlmostEqual(float(result[0, 0]), 25 / 175, places=5)

    def test_id_is_preserved_for_nearby_detection(self):
        tracker = MultiObjectTracker(iou_threshold=0.1, max_age=2)
        first = tracker.update([detection(10, 10, 30, 30)])
        second = tracker.update([detection(12, 11, 32, 31)])
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(second[0].hits, 2)
        self.assertEqual(second[0].state.shape, (8,))

    def test_class_mismatch_creates_new_id(self):
        tracker = MultiObjectTracker(iou_threshold=0.1, max_age=2)
        first = tracker.update([detection(10, 10, 30, 30, class_id=2)])
        second = tracker.update([detection(10, 10, 30, 30, class_id=0)])
        self.assertNotEqual(first[0].track_id, second[0].track_id)

    def test_line_counter_counts_once(self):
        tracker = MultiObjectTracker(iou_threshold=0.1, max_age=2)
        counter = LineCounter(line_ratio=0.5)
        for y in (35, 43, 51, 60, 68):
            tracks = tracker.update([detection(20, y - 10, 40, y + 10)])
            counter.update(tracks, 100)
        self.assertEqual(counter.down, 1)
        self.assertEqual(counter.up, 0)


class OcSortTrackerTests(unittest.TestCase):
    def make_tracker(self, **overrides):
        options = {
            "iou_threshold": 0.2,
            "second_stage_iou": 0.05,
            "max_age": 4,
            "min_hits": 2,
            "high_confidence": 0.5,
            "low_confidence": 0.1,
            "new_track_threshold": 0.5,
        }
        options.update(overrides)
        return OcSortTracker(**options)

    def test_ocsort_uses_official_seven_state_box_parameterization(self):
        tracker = self.make_tracker(min_hits=1)
        visible = tracker.update([detection(10, 20, 30, 60)])
        track = visible[0]
        self.assertIsInstance(track, OcSortTrack)
        self.assertEqual(track.state.shape, (7,))
        self.assertEqual(track.covariance.shape, (7, 7))
        np.testing.assert_allclose(track.state[:4], [20, 40, 800, 0.5])
        np.testing.assert_allclose(track.xyxy, [10, 20, 30, 60], atol=1e-4)

    def test_ocsort_prediction_advances_area_but_not_aspect_ratio(self):
        tracker = self.make_tracker(min_hits=1)
        tracker.update([detection(10, 20, 30, 60)])
        track = tracker.tracks[0]
        track.state[6] = 20.0
        predicted = tracker.predict_only()[0]
        self.assertAlmostEqual(predicted.state[2], 820.0)
        self.assertAlmostEqual(predicted.state[3], 0.5)
        self.assertTrue(np.isfinite(predicted.xyxy).all())

    def test_gui_tracker_choice_uses_seven_state_ocsort_and_eight_state_baseline(self):
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        with patch("mot_core.YoloDetector") as detector_type:
            detector_type.return_value.names = {2: "car"}
            detector_type.return_value.infer.return_value = [detection(10, 10, 30, 30)]
            ocsort = FrameAnalyzer("unused.pt", tracker_type="ocsort")
            baseline = FrameAnalyzer("unused.pt", tracker_type="sort")
            self.assertEqual(ocsort.process(frame).tracks[0].state.shape, (7,))
            self.assertEqual(baseline.process(frame).tracks[0].state.shape, (8,))

    def test_low_score_detection_recovers_confirmed_track(self):
        tracker = self.make_tracker()
        first = tracker.update([detection(10, 10, 30, 30, score=0.9)])
        second = tracker.update([detection(12, 10, 32, 30, score=0.9)])
        recovered = tracker.update([detection(14, 10, 34, 30, score=0.2)])
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(recovered[0].track_id, first[0].track_id)
        self.assertEqual(tracker.summary["low_score_matches"], 1)

    def test_low_score_detection_cannot_create_track(self):
        tracker = self.make_tracker()
        self.assertEqual(tracker.update([detection(10, 10, 30, 30, score=0.2)]), [])
        self.assertEqual(tracker.summary["created_tracks"], 0)

    def test_observation_reupdate_keeps_id_after_gap(self):
        tracker = self.make_tracker(iou_threshold=0.1)
        first = tracker.update([detection(10, 10, 30, 30)])
        tracker.update([detection(12, 10, 32, 30)])
        self.assertEqual(tracker.update([]), [])
        recovered = tracker.update([detection(16, 10, 36, 30)])
        self.assertEqual(recovered[0].track_id, first[0].track_id)
        self.assertEqual(recovered[0].state.shape, (7,))
        self.assertTrue(np.isfinite(recovered[0].state).all())
        self.assertEqual(tracker.summary["oru_updates"], 1)

    def test_class_mismatch_never_reuses_id(self):
        tracker = self.make_tracker(min_hits=1)
        first = tracker.update([detection(10, 10, 30, 30, class_id=2)])
        second = tracker.update([detection(10, 10, 30, 30, class_id=0)])
        self.assertNotEqual(first[0].track_id, second[0].track_id)

    def test_prediction_only_keeps_visible_track_between_detector_frames(self):
        tracker = self.make_tracker(min_hits=1)
        first = tracker.update([detection(10, 10, 30, 30)])
        predicted = tracker.predict_only()
        predicted_again = tracker.predict_only()
        self.assertEqual(predicted[0].track_id, first[0].track_id)
        self.assertEqual(predicted_again[0].track_id, first[0].track_id)
        self.assertEqual(tracker.summary["prediction_only_frames"], 2)

    def test_topic_lite_reid_runs_only_for_person_conflict_after_initialization(self):
        encoder = FakeAppearanceEncoder()
        tracker = self.make_tracker(
            min_hits=1,
            appearance_encoder=encoder,
            reid_ambiguity_margin=1.0,
        )
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        tracker.update([
            detection(0, 0, 20, 30, class_id=0),
            detection(10, 0, 30, 30, class_id=0),
            detection(40, 0, 60, 30, class_id=2),
        ], frame=frame)
        initialization_calls = encoder.calls
        self.assertEqual(initialization_calls, 0)
        self.assertEqual(encoder.crops, 0)

        tracker.update([
            detection(1, 0, 21, 30, class_id=0),
            detection(11, 0, 31, 30, class_id=0),
            detection(40, 0, 60, 30, class_id=2),
        ], frame=frame)
        self.assertGreater(encoder.calls, initialization_calls)
        self.assertGreaterEqual(tracker.summary["reid_conflicts"], 1)
        self.assertEqual(tracker.summary["reid_crops"], encoder.crops)
        self.assertEqual(encoder.crops, 4)

    def test_topic_lite_reid_detects_two_tracks_competing_for_one_detection(self):
        encoder = FakeAppearanceEncoder()
        tracker = self.make_tracker(min_hits=1, appearance_encoder=encoder)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        tracker.update([
            detection(0, 0, 20, 30, class_id=0),
            detection(10, 0, 30, 30, class_id=0),
        ], frame=frame)
        initial_calls = encoder.calls
        tracker.update([detection(5, 0, 25, 30, class_id=0)], frame=frame)
        self.assertGreater(encoder.calls, initial_calls)
        self.assertGreaterEqual(tracker.summary["reid_conflicts"], 1)

    def test_topic_lite_reid_skips_unambiguous_person_matches(self):
        encoder = FakeAppearanceEncoder()
        tracker = self.make_tracker(min_hits=1, appearance_encoder=encoder)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        tracker.update([detection(0, 0, 20, 30, class_id=0)], frame=frame)
        tracker.update([detection(1, 0, 21, 30, class_id=0)], frame=frame)
        self.assertEqual(encoder.calls, 0)
        self.assertEqual(tracker.summary["reid_crops"], 0)

    def test_topic_lite_reid_runs_on_occlusion_recovery(self):
        encoder = FakeAppearanceEncoder()
        tracker = self.make_tracker(min_hits=1, appearance_encoder=encoder)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        first = tracker.update([detection(0, 0, 20, 30, class_id=0)], frame=frame)
        tracker.update([], frame=frame)
        recovered = tracker.update([detection(1, 0, 21, 30, class_id=0)], frame=frame)
        self.assertEqual(recovered[0].track_id, first[0].track_id)
        self.assertEqual(tracker.summary["reid_recoveries"], 1)
        self.assertEqual(encoder.crops, 2)


if __name__ == "__main__":
    unittest.main()
