
"""
Activity analysis for SafeGuard.

Recognizes basic body postures and estimates some activities
using temporal information.

This module does NOT detect falls.
Fall detection remains the responsibility of FallAnalyzer.
"""

from collections import deque
from dataclasses import dataclass, field
import math


# COCO-17 keypoint indices
NOSE = 0
L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12
L_KNEE = 13
R_KNEE = 14
L_ANKLE = 15
R_ANKLE = 16


@dataclass
class ActivityResult:
    """Result returned by ActivityAnalyzer."""

    track_id: int
    activity: str
    posture: str
    reliable: bool
    duration_sec: float
    in_nap_zone: bool
    metrics: dict = field(default_factory=dict)


@dataclass
class _ActivityState:
    """Private temporal memory for one tracked person."""

    history: deque = field(default_factory=deque)
    angle_history: deque = field(default_factory=deque)

    current_posture: str = "unknown"
    posture_since: float | None = None

    last_seen: float | None = None
    lying_since: float | None = None

    movement_times: deque = field(default_factory=deque)
    posture_changes: deque = field(default_factory=deque)


class ActivityAnalyzer:
    """
    Analyze the activity of each tracked person.

    Basic postures:
        standing, sitting, kneeling, bending,
        crawling, lying

    Temporal activity estimates:
        jumping, rolling, sleeping, rough_play

    Input coordinates must be in pixels, with y increasing downward.
    """

    def __init__(
        self,
        min_confidence=0.5,
        lying_angle_deg=55.0,
        upright_angle_deg=30.0,
        sleeping_duration_sec=30.0,
        rough_play_window_sec=5.0,
        rough_play_min_movements=4,
        jump_speed_threshold=1.5,
        rolling_angle_change_deg=45.0,
        max_observation_gap_sec=1.0,
        track_timeout_sec=10.0,
        history_window_sec=5.0,
    ):
        # Validate configuration.
        if not 0 <= min_confidence <= 1:
            raise ValueError(
                "min_confidence must be between 0 and 1"
            )

        if not 0 < upright_angle_deg < lying_angle_deg <= 90:
            raise ValueError(
                "Angle thresholds must satisfy "
                "0 < upright_angle_deg < lying_angle_deg <= 90"
            )

        if sleeping_duration_sec <= 0:
            raise ValueError("sleeping_duration_sec must be positive")

        if rough_play_window_sec <= 0:
            raise ValueError("rough_play_window_sec must be positive")

        if rough_play_min_movements < 1:
            raise ValueError(
                "rough_play_min_movements must be positive"
            )

        if jump_speed_threshold < 0:
            raise ValueError("jump_speed_threshold cannot be negative")

        if rolling_angle_change_deg <= 0:
            raise ValueError(
                "rolling_angle_change_deg must be positive"
            )

        if max_observation_gap_sec <= 0:
            raise ValueError(
                "max_observation_gap_sec must be positive"
            )

        if track_timeout_sec <= 0:
            raise ValueError("track_timeout_sec must be positive")

        if history_window_sec <= 0:
            raise ValueError("history_window_sec must be positive")

        self.min_confidence = min_confidence
        self.lying_angle_deg = lying_angle_deg
        self.upright_angle_deg = upright_angle_deg

        self.sleeping_duration_sec = sleeping_duration_sec

        self.rough_play_window_sec = rough_play_window_sec
        self.rough_play_min_movements = rough_play_min_movements

        self.jump_speed_threshold = jump_speed_threshold
        self.rolling_angle_change_deg = rolling_angle_change_deg

        self.max_observation_gap_sec = max_observation_gap_sec
        self.track_timeout_sec = track_timeout_sec
        self.history_window_sec = history_window_sec

        # Every track ID gets independent activity memory.
        self.people = {}

    # ---------------------------------------------------------
    # KEYPOINT VALIDATION
    # ---------------------------------------------------------

    def _valid_point(self, keypoints, confidence, index):
        """Return a valid (x, y) point or None."""

        if index >= len(keypoints) or index >= len(confidence):
            return None

        point = keypoints[index]

        if point is None or len(point) < 2:
            return None

        try:
            x = float(point[0])
            y = float(point[1])
            conf = float(confidence[index])
        except (TypeError, ValueError, IndexError):
            return None

        if not all(math.isfinite(v) for v in (x, y, conf)):
            return None

        if x <= 0 or y <= 0:
            return None

        if conf < self.min_confidence:
            return None

        return (x, y)

    @staticmethod
    def _midpoint(point_a, point_b):
        """Calculate the midpoint between two points."""

        if point_a is None or point_b is None:
            return None

        return (
            (point_a[0] + point_b[0]) / 2,
            (point_a[1] + point_b[1]) / 2,
        )

    @staticmethod
    def _distance(point_a, point_b):
        """Calculate Euclidean distance between two points."""

        return math.hypot(
            point_a[0] - point_b[0],
            point_a[1] - point_b[1],
        )

    # ---------------------------------------------------------
    # POSE EXTRACTION
    # ---------------------------------------------------------

    def _get_pose(self, keypoints, confidence):
        """
        Extract useful body measurements.

        A valid torso requires both shoulders and both hips.
        Missing nose or ankles do not automatically invalidate it.
        """

        if keypoints is None or confidence is None:
            return None

        if len(keypoints) < 17 or len(confidence) < 17:
            return None

        points = {
            "left_shoulder": self._valid_point(
                keypoints, confidence, L_SHOULDER
            ),
            "right_shoulder": self._valid_point(
                keypoints, confidence, R_SHOULDER
            ),
            "left_hip": self._valid_point(
                keypoints, confidence, L_HIP
            ),
            "right_hip": self._valid_point(
                keypoints, confidence, R_HIP
            ),
            "left_knee": self._valid_point(
                keypoints, confidence, L_KNEE
            ),
            "right_knee": self._valid_point(
                keypoints, confidence, R_KNEE
            ),
            "left_ankle": self._valid_point(
                keypoints, confidence, L_ANKLE
            ),
            "right_ankle": self._valid_point(
                keypoints, confidence, R_ANKLE
            ),
        }

        shoulders = self._midpoint(
            points["left_shoulder"],
            points["right_shoulder"],
        )

        hips = self._midpoint(
            points["left_hip"],
            points["right_hip"],
        )

        if shoulders is None or hips is None:
            return None

        torso_length = self._distance(shoulders, hips)

        if torso_length <= 0:
            return None

        dx = shoulders[0] - hips[0]
        dy = shoulders[1] - hips[1]

        # 0 degrees = upright torso.
        # 90 degrees = horizontal torso.
        torso_angle = math.degrees(
            math.atan2(abs(dx), abs(dy))
        )

        leg_names = (
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        )

        visible_legs = sum(
            points[name] is not None
            for name in leg_names
        )

        knees = [
            points["left_knee"],
            points["right_knee"],
        ]

        knees = [point for point in knees if point is not None]

        ankles = [
            points["left_ankle"],
            points["right_ankle"],
        ]

        ankles = [point for point in ankles if point is not None]

        average_knee_y = (
            sum(point[1] for point in knees) / len(knees)
            if knees else None
        )

        average_ankle_y = (
            sum(point[1] for point in ankles) / len(ankles)
            if ankles else None
        )

        return {
            "shoulders": shoulders,
            "hips": hips,
            "torso_length": torso_length,
            "torso_angle": torso_angle,
            "visible_legs": visible_legs,
            "average_knee_y": average_knee_y,
            "average_ankle_y": average_ankle_y,
        }

    # ---------------------------------------------------------
    # BASIC POSTURE CLASSIFICATION
    # ---------------------------------------------------------

    def _classify_posture(self, pose):
        """Classify one pose using simple geometric rules."""

        angle = pose["torso_angle"]
        legs = pose["visible_legs"]

        hip_y = pose["hips"][1]
        knee_y = pose["average_knee_y"]
        ankle_y = pose["average_ankle_y"]

        # Horizontal torso.
        if angle >= self.lying_angle_deg:

            # Possible crawling:
            # torso is horizontal and knees/ankles
            # appear in a crawling-like arrangement.
            if (
                legs >= 2
                and knee_y is not None
                and ankle_y is not None
                and knee_y > hip_y
                and abs(ankle_y - knee_y)
                < pose["torso_length"] * 1.5
            ):
                return "crawling"

            return "lying"

        # Mostly upright torso.
        if angle <= self.upright_angle_deg:

            if knee_y is None:
                return "standing"

            if ankle_y is not None and knee_y > hip_y:
                return "standing"

            knee_distance = abs(knee_y - hip_y)

            # Approximate sitting rule.
            if knee_distance < pose["torso_length"] * 0.45:
                return "sitting"

            return "kneeling"

        # Between upright and horizontal.
        return "bending"

    # ---------------------------------------------------------
    # TRACK STATE
    # ---------------------------------------------------------

    def _get_state(self, track_id, now):
        """Get or create independent state for a track."""

        state = self.people.get(track_id)

        if state is None:
            state = _ActivityState()
            self.people[track_id] = state
            return state

        # Old track IDs must not inherit stale activity history.
        if (
            state.last_seen is not None
            and now - state.last_seen > self.track_timeout_sec
        ):
            state = _ActivityState()
            self.people[track_id] = state

        return state

    # ---------------------------------------------------------
    # MAIN ANALYSIS
    # ---------------------------------------------------------

    def analyze(
        self,
        track_id,
        keypoints,
        confidence,
        now,
        in_nap_zone=False,
        pose_reliable=True,
    ):
        """
        Analyze one person's activity for the current frame.

        Args:
            track_id: Unique tracking ID.
            keypoints: COCO-17 keypoints in pixel coordinates.
            confidence: Confidence value for each keypoint.
            now: Timestamp in seconds.
            in_nap_zone: Whether the person is inside a nap zone.
            pose_reliable: Result from PoseQualityChecker.

        Returns:
            ActivityResult
        """

        try:
            now = float(now)
        except (TypeError, ValueError):
            raise ValueError("now must be a valid timestamp")

        if not math.isfinite(now):
            raise ValueError("now must be finite")

        state = self._get_state(track_id, now)

        if (
            state.last_seen is not None
            and now < state.last_seen
        ):
            raise ValueError(
                "Timestamps must not move backwards for a track"
            )

        gap = (
            now - state.last_seen
            if state.last_seen is not None
            else None
        )

        state.last_seen = now

        # Long gaps break continuous activity evidence.
        if (
            gap is not None
            and gap > self.max_observation_gap_sec
        ):
            state.history.clear()
            state.angle_history.clear()
            state.movement_times.clear()
            state.posture_changes.clear()

            state.posture_since = None
            state.lying_since = None
            state.current_posture = "unknown"

        if not pose_reliable:
            return ActivityResult(
                track_id=track_id,
                activity="unknown",
                posture="unknown",
                reliable=False,
                duration_sec=0.0,
                in_nap_zone=in_nap_zone,
            )

        pose = self._get_pose(keypoints, confidence)

        if pose is None:
            return ActivityResult(
                track_id=track_id,
                activity="unknown",
                posture="unknown",
                reliable=False,
                duration_sec=0.0,
                in_nap_zone=in_nap_zone,
            )

        posture = self._classify_posture(pose)

        if posture == "unknown":
            return ActivityResult(
                track_id=track_id,
                activity="unknown",
                posture="unknown",
                reliable=False,
                duration_sec=0.0,
                in_nap_zone=in_nap_zone,
            )

        # -----------------------------------------------------
        # POSTURE DURATION
        # -----------------------------------------------------

        previous_posture = state.current_posture

        if posture != previous_posture:
            state.current_posture = posture
            state.posture_since = now
            state.posture_changes.append(now)

        if posture == "lying":
            if state.lying_since is None:
                state.lying_since = now
        else:
            state.lying_since = None

        # -----------------------------------------------------
        # BODY MOVEMENT
        # -----------------------------------------------------

        hip = pose["hips"]
        torso_length = pose["torso_length"]

        normalized_speed = None

        previous_sample = (
            state.history[-1]
            if state.history else None
        )

        if previous_sample is not None:
            old_time, old_x, old_y, old_torso = previous_sample

            dt = now - old_time

            if 0 < dt <= self.max_observation_gap_sec:
                displacement = math.hypot(
                    hip[0] - old_x,
                    hip[1] - old_y,
                )

                reference_length = max(
                    old_torso,
                    torso_length,
                )

                normalized_speed = (
                    displacement / dt / reference_length
                )

                if normalized_speed >= self.jump_speed_threshold:
                    state.movement_times.append(now)

        state.history.append(
            (now, hip[0], hip[1], torso_length)
        )

        # -----------------------------------------------------
        # TORSO ANGLE HISTORY
        # -----------------------------------------------------

        state.angle_history.append(
            (now, pose["torso_angle"])
        )

        # Remove old history.
        while (
            state.history
            and now - state.history[0][0]
            > self.history_window_sec
        ):
            state.history.popleft()

        while (
            state.angle_history
            and now - state.angle_history[0][0]
            > self.history_window_sec
        ):
            state.angle_history.popleft()

        while (
            state.movement_times
            and now - state.movement_times[0]
            > self.rough_play_window_sec
        ):
            state.movement_times.popleft()

        while (
            state.posture_changes
            and now - state.posture_changes[0]
            > self.rough_play_window_sec
        ):
            state.posture_changes.popleft()

        # -----------------------------------------------------
        # DURATION
        # -----------------------------------------------------

        duration = (
            now - state.posture_since
            if state.posture_since is not None
            else 0.0
        )

        activity = posture

        # -----------------------------------------------------
        # SLEEPING
        # -----------------------------------------------------

        if (
            posture == "lying"
            and in_nap_zone
            and state.lying_since is not None
            and now - state.lying_since
            >= self.sleeping_duration_sec
        ):
            activity = "sleeping"

        # -----------------------------------------------------
        # ROLLING
        # -----------------------------------------------------

        angle_change = 0.0

        if len(state.angle_history) >= 2:
            oldest_angle = state.angle_history[0][1]

            angle_change = abs(
                pose["torso_angle"] - oldest_angle
            )

        if (
            posture == "lying"
            and angle_change >= self.rolling_angle_change_deg
        ):
            activity = "rolling"

        # -----------------------------------------------------
        # ROUGH PLAY
        # -----------------------------------------------------

        elif (
            len(state.movement_times)
            >= self.rough_play_min_movements
            and len(state.posture_changes) >= 2
            and posture != "lying"
        ):
            activity = "rough_play"

        # -----------------------------------------------------
        # JUMPING
        # -----------------------------------------------------

        elif (
            normalized_speed is not None
            and normalized_speed >= self.jump_speed_threshold
            and posture == "standing"
        ):
            activity = "jumping"

        return ActivityResult(
            track_id=track_id,
            activity=activity,
            posture=posture,
            reliable=True,
            duration_sec=max(0.0, duration),
            in_nap_zone=in_nap_zone,
            metrics={
                "torso_angle_deg": pose["torso_angle"],
                "normalized_speed": normalized_speed,
                "visible_legs": pose["visible_legs"],
                "angle_change_deg": angle_change,
            },
        )

    # ---------------------------------------------------------
    # OLD TRACK CLEANUP
    # ---------------------------------------------------------

    def forget_old(self, now):
        """Remove tracks that have not been observed recently."""

        now = float(now)

        if not math.isfinite(now):
            raise ValueError("now must be finite")

        expired = [
            track_id
            for track_id, state in self.people.items()
            if (
                state.last_seen is not None
                and now - state.last_seen > self.track_timeout_sec
            )
        ]

        for track_id in expired:
            del self.people[track_id]

        return expired
