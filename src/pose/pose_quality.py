
from dataclasses import dataclass
import math

from src.pose.constants import (
    NOSE,
    L_SHOULDER,
    R_SHOULDER,
    L_HIP,
    R_HIP,
)


@dataclass
class PoseQualityResult:
    reliable: bool
    torso_ok: bool
    nose_ok: bool
    legs_ok: bool
    visible_fraction: float
    overlap: bool
    reason: str


class PoseQualityChecker:
    """
    Checks whether pose keypoints are reliable enough
    to be used by the safety pipeline.

    Expected input:
        keypoints: COCO-17 keypoints in pixel coordinates
        confidence: confidence value for each keypoint
    """

    def __init__(
        self,
        min_confidence=0.5,
        min_visible_fraction=0.4,
    ):
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")

        if not 0 <= min_visible_fraction <= 1:
            raise ValueError("min_visible_fraction must be between 0 and 1")

        self.min_confidence = min_confidence
        self.min_visible_fraction = min_visible_fraction

    def _valid_point(self, keypoints, confidence, index):
        """Check whether one keypoint has valid coordinates and confidence."""

        if index >= len(keypoints) or index >= len(confidence):
            return False

        point = keypoints[index]

        if point is None or len(point) < 2:
            return False

        try:
            x = float(point[0])
            y = float(point[1])
            conf = float(confidence[index])
        except (TypeError, ValueError, IndexError):
            return False

        if not all(math.isfinite(v) for v in (x, y, conf)):
            return False

        if x <= 0 or y <= 0:
            return False

        return conf >= self.min_confidence

    def check(self, keypoints, confidence, overlap=False):
        """
        Return a PoseQualityResult describing pose reliability.

        overlap=True means another person overlaps this person's
        bounding box enough to make the pose ambiguous.
        """

        if keypoints is None or confidence is None:
            return PoseQualityResult(
                reliable=False,
                torso_ok=False,
                nose_ok=False,
                legs_ok=False,
                visible_fraction=0.0,
                overlap=overlap,
                reason="missing_pose",
            )

        total_points = min(len(keypoints), len(confidence))

        if total_points == 0:
            return PoseQualityResult(
                reliable=False,
                torso_ok=False,
                nose_ok=False,
                legs_ok=False,
                visible_fraction=0.0,
                overlap=overlap,
                reason="empty_pose",
            )

        visible_count = sum(
            self._valid_point(keypoints, confidence, i)
            for i in range(total_points)
        )

        visible_fraction = visible_count / 17.0

        # Both shoulders and both hips are required
        # for a reliable torso measurement.
        torso_indices = (
            L_SHOULDER,
            R_SHOULDER,
            L_HIP,
            R_HIP,
        )

        torso_ok = all(
            self._valid_point(keypoints, confidence, index)
            for index in torso_indices
        )

        nose_ok = self._valid_point(
            keypoints, confidence, NOSE
        )

        # At least two lower-body points, including an ankle,
        # are required to say that the legs are sufficiently visible.
        leg_indices = (13, 14, 15, 16)

        visible_legs = sum(
            self._valid_point(keypoints, confidence, index)
            for index in leg_indices
        )

        legs_ok = visible_legs >= 2 and any(
            self._valid_point(keypoints, confidence, index)
            for index in (15, 16)
        )

        reasons = []

        if visible_fraction < self.min_visible_fraction:
            reasons.append("too_few_visible_keypoints")

        if not torso_ok:
            reasons.append("unreliable_torso")

        if overlap:
            reasons.append("person_overlap")

        reliable = (
            visible_fraction >= self.min_visible_fraction
            and torso_ok
            and not overlap
        )

        if reliable:
            reason = "reliable"
        else:
            reason = ", ".join(reasons)

        return PoseQualityResult(
            reliable=reliable,
            torso_ok=torso_ok,
            nose_ok=nose_ok,
            legs_ok=legs_ok,
            visible_fraction=visible_fraction,
            overlap=overlap,
            reason=reason,
        )
