import unittest

import numpy as np

from mot_core import Detection, LineCounter, MultiObjectTracker, OcSortTracker, iou_matrix


def detection(x1, y1, x2, y2, class_id=2, score=0.9):
    return Detection(np.array([x1, y1, x2, y2], dtype=np.float32), score, class_id)


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
        self.assertEqual(tracker.summary["oru_updates"], 1)

    def test_class_mismatch_never_reuses_id(self):
        tracker = self.make_tracker(min_hits=1)
        first = tracker.update([detection(10, 10, 30, 30, class_id=2)])
        second = tracker.update([detection(10, 10, 30, 30, class_id=0)])
        self.assertNotEqual(first[0].track_id, second[0].track_id)


if __name__ == "__main__":
    unittest.main()
