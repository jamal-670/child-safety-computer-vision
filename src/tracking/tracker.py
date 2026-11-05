"""Reusable adapter for Ultralytics tracking results.

The adapter exposes pose detections with their tracker IDs without making any
event-specific decisions. Missing IDs stay ``None`` rather than being presented
as stable identities.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrackedPerson:
    """Pose information for one person in a frame."""

    track_id: int | None
    keypoints: Any
    confidence: Any | None


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class ByteTrackAdapter:
    """Call a model's Ultralytics ``track`` API and align person pose data."""

    def __init__(self, model, confidence, tracker="bytetrack.yaml"):
        self.model = model
        self.confidence = confidence
        self.tracker = tracker

    def track(self, frame):
        """Return detections in model order, preserving IDs when available."""
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        keypoints_result = getattr(result, "keypoints", None)
        xy = _to_numpy(
            getattr(keypoints_result, "xy", None)
            if keypoints_result is not None
            else None
        )
        if xy is None:
            return []

        confidences = _to_numpy(
            getattr(keypoints_result, "conf", None)
            if keypoints_result is not None
            else None
        )
        if confidences is not None and len(confidences) != len(xy):
            confidences = None

        boxes = getattr(result, "boxes", None)
        raw_ids = _to_numpy(getattr(boxes, "id", None) if boxes is not None else None)
        ids = None if raw_ids is None else raw_ids.reshape(-1).tolist()
        if ids is not None and len(ids) != len(xy):
            ids = None

        people = []
        for index, person_keypoints in enumerate(xy):
            track_id = None
            if ids is not None and ids[index] is not None:
                try:
                    track_id = int(ids[index])
                except (TypeError, ValueError, OverflowError):
                    track_id = None
            person_confidence = (
                confidences[index] if confidences is not None else None
            )
            people.append(
                TrackedPerson(track_id, person_keypoints, person_confidence)
            )
        return people
