"""Core detector, tracker and visualization logic for the interactive MOT demo.

The GUI imports this module, while the smoke test and unit tests can use it
without creating a window.  Tracking is intentionally implemented here rather
than delegated to Ultralytics so the data association and lifecycle are easy to
inspect and explain.
"""

from __future__ import annotations

import colorsys
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("YOLO_VERBOSE", "False")

TRAFFIC_CLASS_IDS = {0, 1, 2, 3, 5, 7}


@dataclass(frozen=True)
class Detection:
    xyxy: np.ndarray
    score: float
    class_id: int


@dataclass
class Track:
    track_id: int
    class_id: int
    score: float
    state: np.ndarray
    covariance: np.ndarray
    hits: int = 1
    age: int = 1
    missed: int = 0
    history: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=64))
    last_observation: Detection | None = None
    observations: dict[int, Detection] = field(default_factory=dict)
    velocity: np.ndarray | None = None
    observed: bool = True
    frozen_state: np.ndarray | None = None
    frozen_covariance: np.ndarray | None = None
    _pre_predict_state: np.ndarray | None = None
    _pre_predict_covariance: np.ndarray | None = None

    @staticmethod
    def _measurement(box: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(float, box)
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float32)

    @classmethod
    def create(cls, track_id: int, detection: Detection, trail_length: int) -> "Track":
        state = np.zeros(8, dtype=np.float32)
        state[:4] = cls._measurement(detection.xyxy)
        track = cls(
            track_id=track_id,
            class_id=detection.class_id,
            score=detection.score,
            state=state,
            covariance=np.diag([10, 10, 10, 10, 100, 100, 100, 100]).astype(np.float32),
            history=deque(maxlen=trail_length),
        )
        track.history.append(track.center)
        track.last_observation = detection
        track.observations[0] = detection
        return track

    @property
    def center(self) -> tuple[int, int]:
        return int(self.state[0]), int(self.state[1])

    @property
    def xyxy(self) -> np.ndarray:
        cx, cy, width, height = self.state[:4]
        width = max(2.0, float(width))
        height = max(2.0, float(height))
        return np.array(
            [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
            dtype=np.float32,
        )

    def predict(self) -> None:
        transition = np.eye(8, dtype=np.float32)
        transition[0, 4] = transition[1, 5] = 1.0
        transition[2, 6] = transition[3, 7] = 1.0
        process_noise = np.diag([1, 1, 1, 1, 4, 4, 4, 4]).astype(np.float32)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self.state[2:4] = np.maximum(self.state[2:4], 2.0)
        self.age += 1
        self.missed += 1

    def _correct(self, box: Sequence[float]) -> None:
        measurement = self._measurement(box)
        observation = np.zeros((4, 8), dtype=np.float32)
        observation[:4, :4] = np.eye(4, dtype=np.float32)
        noise = np.diag([4, 4, 9, 9]).astype(np.float32)
        innovation = measurement - observation @ self.state
        innovation_cov = observation @ self.covariance @ observation.T + noise
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(8, dtype=np.float32) - gain @ observation) @ self.covariance

    def update(self, detection: Detection) -> None:
        self._correct(detection.xyxy)
        self.class_id = detection.class_id
        self.score = detection.score
        self.hits += 1
        self.missed = 0
        self.history.append(self.center)
        self.last_observation = detection
        self.observations[self.age - 1] = detection

    def oc_predict(self) -> None:
        if self.observed:
            self._pre_predict_state = self.state.copy()
            self._pre_predict_covariance = self.covariance.copy()
        self.predict()

    def previous_observation(self, delta_t: int) -> Detection | None:
        current_frame = self.age - 1
        for offset in range(delta_t, 0, -1):
            observation = self.observations.get(current_frame - offset)
            if observation is not None:
                return observation
        if not self.observations:
            return None
        return self.observations[max(self.observations)]

    def mark_oc_missed(self) -> None:
        if self.observed and self._pre_predict_state is not None:
            self.frozen_state = self._pre_predict_state.copy()
            self.frozen_covariance = self._pre_predict_covariance.copy()
        self.observed = False

    def update_oc(self, detection: Detection, delta_t: int) -> bool:
        previous = self.previous_observation(delta_t)
        if previous is not None:
            self.velocity = box_direction(previous.xyxy, detection.xyxy)

        reupdated = (
            not self.observed
            and self.frozen_state is not None
            and self.frozen_covariance is not None
            and self.last_observation is not None
        )
        if reupdated:
            gap = max(1, self.missed)
            first = np.asarray(self.last_observation.xyxy, dtype=np.float32)
            second = np.asarray(detection.xyxy, dtype=np.float32)
            self.state = self.frozen_state.copy()
            self.covariance = self.frozen_covariance.copy()
            for step in range(1, gap + 1):
                transition = np.eye(8, dtype=np.float32)
                transition[0, 4] = transition[1, 5] = 1.0
                transition[2, 6] = transition[3, 7] = 1.0
                process_noise = np.diag([1, 1, 1, 1, 4, 4, 4, 4]).astype(np.float32)
                self.state = transition @ self.state
                self.covariance = transition @ self.covariance @ transition.T + process_noise
                virtual_box = first + (second - first) * (step / gap)
                self._correct(virtual_box)
        else:
            self._correct(detection.xyxy)

        self.class_id = detection.class_id
        self.score = detection.score
        self.hits += 1
        self.missed = 0
        self.observed = True
        self.last_observation = detection
        self.observations[self.age - 1] = detection
        self.history.append(self.center)
        self.frozen_state = None
        self.frozen_covariance = None
        return reupdated


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Return pairwise IoU for two xyxy arrays."""
    boxes_a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.clip(boxes_a[:, 2:] - boxes_a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(boxes_b[:, 2:] - boxes_b[:, :2], 0, None), axis=1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection + 1e-7)


def box_direction(first: Sequence[float], second: Sequence[float]) -> np.ndarray | None:
    """Return the normalized center motion from the first xyxy box to the second."""
    first_box = np.asarray(first, dtype=np.float32)
    second_box = np.asarray(second, dtype=np.float32)
    first_center = np.array([(first_box[0] + first_box[2]) * 0.5,
                             (first_box[1] + first_box[3]) * 0.5], dtype=np.float32)
    second_center = np.array([(second_box[0] + second_box[2]) * 0.5,
                              (second_box[1] + second_box[3]) * 0.5], dtype=np.float32)
    motion = second_center - first_center
    length = float(np.linalg.norm(motion))
    if length <= 1e-6:
        return None
    return motion / length


class MultiObjectTracker:
    """Constant-velocity Kalman tracker with class-aware Hungarian matching."""

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 20, min_hits: int = 1,
                 trail_length: int = 64) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.trail_length = trail_length
        self.tracks: list[Track] = []
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        for track in self.tracks:
            track.predict()

        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))

        if self.tracks and detections:
            predicted = np.stack([track.xyxy for track in self.tracks])
            observed = np.stack([detection.xyxy for detection in detections])
            overlaps = iou_matrix(predicted, observed)
            cost = 1.0 - overlaps
            for track_index, track in enumerate(self.tracks):
                for detection_index, detection in enumerate(detections):
                    if track.class_id != detection.class_id:
                        cost[track_index, detection_index] = 1e6
            row_indices, column_indices = linear_sum_assignment(cost)
            for track_index, detection_index in zip(row_indices, column_indices):
                if cost[track_index, detection_index] < 1e5 and overlaps[track_index, detection_index] >= self.iou_threshold:
                    matches.append((int(track_index), int(detection_index)))
                    unmatched_tracks.discard(int(track_index))
                    unmatched_detections.discard(int(detection_index))

        for track_index, detection_index in matches:
            self.tracks[track_index].update(detections[detection_index])

        for detection_index in sorted(unmatched_detections):
            self.tracks.append(Track.create(self.next_id, detections[detection_index], self.trail_length))
            self.next_id += 1

        self.tracks = [track for track in self.tracks if track.missed <= self.max_age]
        return [track for track in self.tracks if track.missed == 0 and track.hits >= self.min_hits]


class OcSortTracker:
    """Class-aware motion-only OC-SORT with OCM, OCR, ORU and low-score recovery."""

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 20, min_hits: int = 2,
                 trail_length: int = 64, high_confidence: float = 0.35,
                 low_confidence: float = 0.10, second_stage_iou: float = 0.10,
                 delta_t: int = 3, inertia: float = 0.20,
                 new_track_threshold: float | None = None) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.trail_length = trail_length
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence
        self.second_stage_iou = second_stage_iou
        self.delta_t = delta_t
        self.inertia = inertia
        self.new_track_threshold = high_confidence if new_track_threshold is None else new_track_threshold
        if not (0.0 <= self.low_confidence < self.high_confidence <= self.new_track_threshold <= 1.0):
            raise ValueError("OC-SORT confidence thresholds must satisfy low < high <= new <= 1")
        if not (0.0 < self.iou_threshold <= 1.0 and 0.0 < self.second_stage_iou <= 1.0):
            raise ValueError("OC-SORT IoU thresholds must be in (0, 1]")
        if self.max_age < 1 or self.min_hits < 1 or self.delta_t < 1 or not (0.0 <= self.inertia <= 1.0):
            raise ValueError("Invalid OC-SORT lifecycle or momentum options")
        self.tracks: list[Track] = []
        self.next_id = 1
        self.frame_count = 0
        self.summary = {
            "ocm_matches": 0,
            "ocr_matches": 0,
            "low_score_matches": 0,
            "oru_updates": 0,
            "created_tracks": 0,
        }

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        for key in self.summary:
            self.summary[key] = 0

    def _associate(self, track_indices: Sequence[int], detection_indices: Sequence[int],
                   detections: Sequence[Detection], track_boxes: Sequence[np.ndarray],
                   threshold: float, use_momentum: bool = False) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        track_indices = list(track_indices)
        detection_indices = list(detection_indices)
        if not track_indices:
            return [], [], detection_indices
        if not detection_indices:
            return [], track_indices, []

        boxes_a = np.stack([track_boxes[index] for index in track_indices])
        boxes_b = np.stack([detections[index].xyxy for index in detection_indices])
        overlaps = iou_matrix(boxes_a, boxes_b)
        quality = overlaps.astype(np.float64)
        valid = np.ones_like(quality, dtype=bool)
        for row, track_index in enumerate(track_indices):
            track = self.tracks[track_index]
            previous = track.previous_observation(self.delta_t) if use_momentum else None
            for column, detection_index in enumerate(detection_indices):
                detection = detections[detection_index]
                if track.class_id != detection.class_id:
                    valid[row, column] = False
                    continue
                if use_momentum and track.velocity is not None and previous is not None:
                    candidate = box_direction(previous.xyxy, detection.xyxy)
                    if candidate is not None:
                        cosine = float(np.clip(np.dot(track.velocity, candidate), -1.0, 1.0))
                        angle = float(np.arccos(cosine))
                        consistency = (np.pi * 0.5 - abs(angle)) / np.pi
                        quality[row, column] += self.inertia * consistency * detection.score

        cost = -quality
        cost[~valid] = 1e6
        rows, columns = linear_sum_assignment(cost)
        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for row, column in zip(rows, columns):
            if valid[row, column] and overlaps[row, column] >= threshold:
                track_index = track_indices[int(row)]
                detection_index = detection_indices[int(column)]
                matches.append((track_index, detection_index))
                matched_tracks.add(track_index)
                matched_detections.add(detection_index)
        return (
            matches,
            [index for index in track_indices if index not in matched_tracks],
            [index for index in detection_indices if index not in matched_detections],
        )

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        selected = [detection for detection in detections if detection.score >= self.low_confidence]
        high_indices = [index for index, detection in enumerate(selected)
                        if detection.score >= self.high_confidence]
        low_indices = [index for index, detection in enumerate(selected)
                       if self.low_confidence <= detection.score < self.high_confidence]

        for track in self.tracks:
            track.oc_predict()
        predicted_boxes = [track.xyxy for track in self.tracks]
        all_track_indices = list(range(len(self.tracks)))

        ocm, unmatched_tracks, unmatched_high = self._associate(
            all_track_indices, high_indices, selected, predicted_boxes,
            self.iou_threshold, use_momentum=True,
        )
        self.summary["ocm_matches"] += len(ocm)

        last_boxes = [track.last_observation.xyxy if track.last_observation is not None else track.xyxy
                      for track in self.tracks]
        ocr, unmatched_tracks, unmatched_high = self._associate(
            unmatched_tracks, unmatched_high, selected, last_boxes,
            self.iou_threshold,
        )
        self.summary["ocr_matches"] += len(ocr)

        low_recovery_tracks = [index for index in unmatched_tracks
                               if self.tracks[index].hits >= self.min_hits
                               and self.tracks[index].missed == 1]
        low_matches, _, _ = self._associate(
            low_recovery_tracks, low_indices, selected, predicted_boxes,
            self.second_stage_iou,
        )
        self.summary["low_score_matches"] += len(low_matches)

        matches = ocm + ocr + low_matches
        matched_track_indices = {track_index for track_index, _ in matches}
        matched_detection_indices = {detection_index for _, detection_index in matches}
        for track_index, detection_index in matches:
            if self.tracks[track_index].update_oc(selected[detection_index], self.delta_t):
                self.summary["oru_updates"] += 1

        for track_index, track in enumerate(self.tracks):
            if track_index not in matched_track_indices:
                track.mark_oc_missed()

        for detection_index in high_indices:
            if detection_index in matched_detection_indices:
                continue
            detection = selected[detection_index]
            if detection.score >= self.new_track_threshold:
                self.tracks.append(Track.create(self.next_id, detection, self.trail_length))
                self.next_id += 1
                self.summary["created_tracks"] += 1

        self.tracks = [track for track in self.tracks if track.missed <= self.max_age]
        self.frame_count += 1
        return [track for track in self.tracks
                if track.missed == 0
                and (track.hits >= self.min_hits or self.frame_count <= self.min_hits)]


class LineCounter:
    def __init__(self, line_ratio: float = 0.62) -> None:
        self.line_ratio = line_ratio
        self.up = 0
        self.down = 0
        self.counted_ids: set[int] = set()

    def reset(self) -> None:
        self.up = self.down = 0
        self.counted_ids.clear()

    def update(self, tracks: Iterable[Track], frame_height: int) -> None:
        line_y = int(frame_height * self.line_ratio)
        for track in tracks:
            if track.track_id in self.counted_ids or len(track.history) < 2:
                continue
            previous_y = track.history[-2][1]
            current_y = track.history[-1][1]
            if previous_y < line_y <= current_y and current_y - previous_y >= 2:
                self.down += 1
                self.counted_ids.add(track.track_id)
            elif previous_y > line_y >= current_y and previous_y - current_y >= 2:
                self.up += 1
                self.counted_ids.add(track.track_id)


class YoloDetector:
    """Lazy wrapper around the repository's bundled Ultralytics model."""

    def __init__(self, model_path: str | Path, device: str = "auto") -> None:
        from ultralytics import YOLO

        import torch

        self.device = 0 if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)
        self.names = self.model.model.names

    def infer(self, frame: np.ndarray, confidence: float, nms_iou: float,
              traffic_only: bool = True) -> list[Detection]:
        result = self.model.predict(
            source=frame,
            conf=confidence,
            iou=nms_iou,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        detections = [Detection(box, float(score), int(class_id))
                      for box, score, class_id in zip(boxes, scores, class_ids)]
        if traffic_only:
            detections = [detection for detection in detections if detection.class_id in TRAFFIC_CLASS_IDS]
        return detections


@dataclass
class FrameResult:
    frame: np.ndarray
    tracks: list[Track]
    detections: int
    up_count: int
    down_count: int
    inference_ms: float


class FrameAnalyzer:
    def __init__(self, model_path: str | Path, confidence: float = 0.35, nms_iou: float = 0.55,
                 track_iou: float = 0.25, max_age: int = 20, trail_length: int = 64,
                 line_ratio: float = 0.62, traffic_only: bool = True, device: str = "auto",
                 tracker_type: str = "ocsort", low_confidence: float = 0.10) -> None:
        self.detector = YoloDetector(model_path, device=device)
        self.tracker_type = tracker_type.lower()
        self.low_confidence = min(low_confidence, max(0.01, confidence - 0.05))
        if self.tracker_type == "ocsort":
            self.tracker = OcSortTracker(
                track_iou,
                max_age,
                trail_length=trail_length,
                high_confidence=confidence,
                low_confidence=self.low_confidence,
                second_stage_iou=min(0.10, track_iou),
            )
        elif self.tracker_type == "sort":
            self.tracker = MultiObjectTracker(track_iou, max_age, trail_length=trail_length)
        else:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")
        self.counter = LineCounter(line_ratio)
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.traffic_only = traffic_only

    def reset(self) -> None:
        self.tracker.reset()
        self.counter.reset()

    def process(self, frame: np.ndarray, selected_id: int | None = None) -> FrameResult:
        started = time.perf_counter()
        detector_confidence = self.low_confidence if self.tracker_type == "ocsort" else self.confidence
        detections = self.detector.infer(frame, detector_confidence, self.nms_iou, self.traffic_only)
        tracks = self.tracker.update(detections)
        self.counter.update(tracks, frame.shape[0])
        inference_ms = (time.perf_counter() - started) * 1000
        annotated = draw_tracking_overlay(
            frame,
            tracks,
            self.detector.names,
            self.counter,
            selected_id=selected_id,
        )
        return FrameResult(annotated, tracks, len(detections), self.counter.up, self.counter.down, inference_ms)


def color_for_id(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    return int(blue * 255), int(green * 255), int(red * 255)


def draw_tracking_overlay(frame: np.ndarray, tracks: Sequence[Track], names: dict | list,
                          counter: LineCounter, selected_id: int | None = None) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    line_y = int(height * counter.line_ratio)
    cv2.line(output, (0, line_y), (width, line_y), (0, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(output, f"COUNT LINE  UP {counter.up}  DOWN {counter.down}", (18, max(28, line_y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 210, 255), 2, cv2.LINE_AA)

    for track in tracks:
        x1, y1, x2, y2 = track.xyxy.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        color = color_for_id(track.track_id)
        thickness = 4 if track.track_id == selected_id else 2
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        class_name = names[track.class_id] if isinstance(names, (dict, list)) else str(track.class_id)
        label = f"ID {track.track_id} | {class_name} {track.score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        top = max(0, y1 - text_height - 10)
        cv2.rectangle(output, (x1, top), (min(width - 1, x1 + text_width + 10), y1), color, -1)
        cv2.putText(output, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (20, 24, 30), 1, cv2.LINE_AA)
        points = list(track.history)
        for index in range(1, len(points)):
            trail_thickness = max(1, int(5 * (1 - index / max(1, len(points)))))
            cv2.line(output, points[index - 1], points[index], color, trail_thickness, cv2.LINE_AA)
        cv2.circle(output, track.center, 4, color, -1, cv2.LINE_AA)
    return output


def read_image(path: str | Path) -> np.ndarray | None:
    """Unicode-safe image read on Windows."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def write_image(path: str | Path, image: np.ndarray) -> bool:
    suffix = Path(path).suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True
