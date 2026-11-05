"""Fall detection from reliable pose measurements for each tracked person.

A possible fall requires a fast downward movement followed by a horizontal pose.
Confirmation counts time only between reliable, nearby observations of that pose.
These pose heuristics are evidence, not a perfect classifier for crawling,
playing, rolling, sleeping, or falls.
"""

import math
from collections import deque

from src.pose.constants import (
    L_HIP,
    L_SHOULDER,
    NOSE,
    R_HIP,
    R_SHOULDER,
)
from src.utils import geometry as geo


_REQUIRED_SETTINGS = (
    "min_joint_conf",
    "smooth_frames",
    "angle_deg",
    "recover_angle_deg",
    "aspect_ratio",
    "speed_window_sec",
    "drop_speed",
    "drop_window_sec",
    "stay_down_sec",
    "possible_timeout_sec",
    "cooldown_sec",
    "track_timeout_sec",
)


def _fmt(value, digits=1):
    return "--" if value is None else f"{value:.{digits}f}"


def _validate_config(cfg):
    """Reject missing or invalid thresholds instead of silently ignoring them."""
    missing = [key for key in _REQUIRED_SETTINGS if key not in cfg]
    if missing:
        raise ValueError(f"Missing fall configuration values: {', '.join(missing)}")

    smooth_frames = cfg["smooth_frames"]
    if isinstance(smooth_frames, bool) or not isinstance(smooth_frames, int):
        raise ValueError("fall.smooth_frames must be a positive integer")
    if smooth_frames < 1:
        raise ValueError("fall.smooth_frames must be a positive integer")

    numeric_settings = (
        key for key in _REQUIRED_SETTINGS
        if key != "smooth_frames"
    )
    for key in numeric_settings:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"fall.{key} must be a finite number")
        if not math.isfinite(value):
            raise ValueError(f"fall.{key} must be a finite number")

    for key, default in (
        ("min_torso_length_px", 10),
        ("min_speed_interval_sec", 0.05),
        ("max_observation_gap_sec", 1.0),
    ):
        value = cfg.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"fall.{key} must be a positive finite number")

    if not 0 <= cfg["min_joint_conf"] <= 1:
        raise ValueError("fall.min_joint_conf must be between 0 and 1")
    if not 0 < cfg["recover_angle_deg"] < cfg["angle_deg"] <= 90:
        raise ValueError(
            "fall angles must satisfy 0 < recover_angle_deg < angle_deg <= 90"
        )
    if cfg["aspect_ratio"] <= 0:
        raise ValueError("fall.aspect_ratio must be positive")
    if cfg["possible_timeout_sec"] < cfg["stay_down_sec"]:
        raise ValueError(
            "fall.possible_timeout_sec must be at least stay_down_sec"
        )
    for key in (
        "speed_window_sec",
        "min_torso_length_px",
        "min_speed_interval_sec",
        "drop_window_sec",
        "stay_down_sec",
        "possible_timeout_sec",
        "track_timeout_sec",
    ):
        if cfg[key] <= 0:
            raise ValueError(f"fall.{key} must be positive")
    if cfg["cooldown_sec"] < 0 or cfg["drop_speed"] < 0:
        raise ValueError("fall.cooldown_sec and fall.drop_speed cannot be negative")


class FallAnalyzer:
    """Track fall-detection state independently for each person ID."""

    def __init__(self, cfg):
        _validate_config(cfg)
        self.cfg = dict(cfg)
        self.cfg.setdefault("min_torso_length_px", 10)
        self.cfg.setdefault("min_speed_interval_sec", 0.05)
        self.cfg.setdefault("max_observation_gap_sec", 1.0)
        self.people = {}

    def _new_memory(self):
        return {
            # Every sample is (timestamp, x, y, torso_length).
            "hip_hist": deque(),
            "chest_hist": deque(),
            "head_hist": deque(),
            "angles": deque(maxlen=self.cfg["smooth_frames"]),
            "state": "normal",
            "possible_since": None,
            "possible_observed_seconds": 0.0,
            "last_possible_observation": None,
            "down_seconds": 0.0,
            "last_lying_observation": None,
            "last_drop_time": None,
            "last_alert_time": None,
            "last_seen": None,
            "last_reliable_time": None,
        }

    def _append_sample(self, history, now, point, torso_len):
        if point is not None:
            if (
                history
                and now - history[-1][0] > self.cfg["max_observation_gap_sec"]
            ):
                history.clear()
            history.append((now, point[0], point[1], torso_len))
            while history and now - history[0][0] > self.cfg["speed_window_sec"]:
                history.popleft()

    def _vertical_hip_speed(self, history, now):
        """Return downward hip speed normalized by mean torso length."""
        return self._normalized_body_speed(history, now, vertical_only=True)

    def _body_point_speed(self, history, now, vertical_only=False):
        """Return vertical or overall body-point speed in torso lengths/second."""
        return self._normalized_body_speed(history, now, vertical_only)

    def _normalized_body_speed(self, history, now, vertical_only):
        if len(history) < 2 or history[-1][0] != now:
            return None

        first = history[0]
        previous = history[-2]
        last = history[-1]
        dt = now - first[0]
        if (
            dt <= self.cfg["min_speed_interval_sec"]
            or now - previous[0] <= self.cfg["min_speed_interval_sec"]
            or dt > self.cfg["max_observation_gap_sec"]
        ):
            return None

        mean_torso = sum(sample[3] for sample in history) / len(history)
        if not math.isfinite(mean_torso) or mean_torso <= 0:
            return None

        displacement = (
            last[2] - first[2]
            if vertical_only
            else math.hypot(last[1] - first[1], last[2] - first[2])
        )
        return displacement / dt / mean_torso

    def analyze_person(self, track_id, kpts, confs, now):
        """Analyze a COCO-17 pose at timestamp ``now`` (seconds)."""
        if isinstance(now, bool) or not isinstance(now, (int, float)):
            raise ValueError("now must be a finite timestamp")
        if not math.isfinite(now):
            raise ValueError("now must be a finite timestamp")
        cfg = self.cfg
        p = self.people.setdefault(track_id, self._new_memory())
        if p["last_seen"] is not None and now < p["last_seen"]:
            raise ValueError("now must not move backwards for a track")
        if (
            p["last_seen"] is not None
            and now - p["last_seen"] > cfg["track_timeout_sec"]
        ):
            # A tracker may recycle an ID after it has gone quiet. Do not let
            # the new appearance inherit the previous person's fall state.
            p = self._new_memory()
            self.people[track_id] = p
        p["last_seen"] = now
        points = [
            geo.get_point(kpts, confs, i, cfg["min_joint_conf"])
            for i in range(17)
        ]

        shoulder_count = sum(points[i] is not None for i in (L_SHOULDER, R_SHOULDER))
        hip_count = sum(points[i] is not None for i in (L_HIP, R_HIP))
        shoulder_mid = geo.midpoint(points[L_SHOULDER], points[R_SHOULDER])
        hip_mid = geo.midpoint(points[L_HIP], points[R_HIP])
        angle = (
            geo.torso_angle(shoulder_mid, hip_mid)
            if shoulder_count and hip_count
            else None
        )
        torso_len = (
            geo.distance(shoulder_mid, hip_mid)
            if shoulder_count and hip_count
            else None
        )

        # A single visible joint can provide a rough angle, but not a stable
        # movement baseline: the visible left/right side may change by frame.
        torso_reliable = (
            shoulder_count == 2
            and hip_count == 2
            and angle is not None
            and torso_len is not None
            and torso_len >= cfg["min_torso_length_px"]
        )
        if torso_reliable:
            last_reliable = p["last_reliable_time"]
            if (
                last_reliable is not None
                and now - last_reliable > cfg["max_observation_gap_sec"]
            ):
                p["angles"].clear()
                p["hip_hist"].clear()
                p["chest_hist"].clear()
                p["head_hist"].clear()
                p["last_drop_time"] = None
                p["down_seconds"] = 0.0
                p["last_lying_observation"] = None
            p["last_reliable_time"] = now

        body_indexes = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, 13, 14, 15, 16)
        body_points = [points[index] for index in body_indexes]
        body_point_count = sum(point is not None for point in body_points)
        ratio = geo.aspect_ratio(body_points)

        if torso_reliable:
            p["angles"].append(angle)
            self._append_sample(p["hip_hist"], now, hip_mid, torso_len)
            self._append_sample(p["chest_hist"], now, shoulder_mid, torso_len)
            self._append_sample(p["head_hist"], now, points[NOSE], torso_len)

        hip_speed = self._vertical_hip_speed(p["hip_hist"], now)
        chest_speed = self._body_point_speed(p["chest_hist"], now)
        head_speed = self._body_point_speed(p["head_hist"], now)
        chest_down_speed = self._body_point_speed(
            p["chest_hist"], now, vertical_only=True
        )
        head_down_speed = self._body_point_speed(
            p["head_hist"], now, vertical_only=True
        )

        smooth_angle = (
            sum(p["angles"]) / len(p["angles"]) if p["angles"] else None
        )
        aspect_support = ratio is not None and ratio > cfg["aspect_ratio"]
        # Torso angle is primary. A wide body shape can support a borderline
        # angle, but the ratio can never identify a lying pose by itself.
        lying = (
            torso_reliable
            and body_point_count >= 4
            and smooth_angle is not None
            and (
                smooth_angle > cfg["angle_deg"]
                or (
                    aspect_support
                    and smooth_angle > cfg["recover_angle_deg"]
                )
            )
        )
        recovered = (
            torso_reliable
            and smooth_angle is not None
            # Upright torso evidence alone is sufficient for recovery.
            and smooth_angle < cfg["recover_angle_deg"]
        )

        # Hip movement is primary. A fast chest drop can support it when the
        # hip estimate is weak; a visible nose corroborates chest movement,
        # but an unavailable nose does not block the chest signal.
        hip_drop = hip_speed is not None and hip_speed > cfg["drop_speed"]
        upper_body_drop = (
            chest_down_speed is not None
            and chest_down_speed > cfg["drop_speed"]
            and (
                head_down_speed is None
                or head_down_speed > cfg["drop_speed"]
            )
        )
        if hip_drop or upper_body_drop:
            p["last_drop_time"] = now
        recent_drop = (
            p["last_drop_time"] is not None
            and 0 <= now - p["last_drop_time"] <= cfg["drop_window_sec"]
        )

        new_alert = False
        if p["state"] == "normal":
            if lying and recent_drop:
                p["state"] = "possible"
                p["possible_since"] = now
                p["possible_observed_seconds"] = 0.0
                p["last_possible_observation"] = now
                p["down_seconds"] = 0.0
                p["last_lying_observation"] = now
        elif p["state"] == "possible":
            if torso_reliable:
                previous_observation = p["last_possible_observation"]
                observation_gap = (
                    now - previous_observation
                    if previous_observation is not None
                    else None
                )
                if (
                    observation_gap is not None
                    and 0 <= observation_gap <= cfg["max_observation_gap_sec"]
                ):
                    p["possible_observed_seconds"] += observation_gap
                else:
                    # Long gaps contribute neither event age nor lying time.
                    p["down_seconds"] = 0.0
                    p["last_lying_observation"] = None
                p["last_possible_observation"] = now

                if recovered:
                    self._reset_possible(p)
                elif lying:
                    if (
                        p["last_lying_observation"] is not None
                        and observation_gap is not None
                        and 0 <= observation_gap <= cfg["max_observation_gap_sec"]
                    ):
                        p["down_seconds"] += observation_gap
                    else:
                        p["down_seconds"] = 0.0
                    p["last_lying_observation"] = now

                    if p["down_seconds"] >= cfg["stay_down_sec"]:
                        p["state"] = "alerted"
                        last_alert = p["last_alert_time"]
                        if (
                            last_alert is None
                            or now - last_alert >= cfg["cooldown_sec"]
                        ):
                            new_alert = True
                            p["last_alert_time"] = now
                else:
                    # A reliable non-lying pose breaks continuous down-time.
                    p["down_seconds"] = 0.0
                    p["last_lying_observation"] = None

                # The timeout advances only while reliable torso observations
                # arrive. A timeout equal to stay_down_sec still permits
                # confirmation on the observation that reaches the threshold.
                if (
                    p["state"] == "possible"
                    and p["possible_observed_seconds"]
                    >= cfg["possible_timeout_sec"]
                ):
                    self._reset_possible(p)
        elif p["state"] == "alerted" and recovered:
            # Motion while still fallen does not cancel an already raised alert.
            p["state"] = "normal"
            p["possible_since"] = None
            p["possible_observed_seconds"] = 0.0
            p["last_possible_observation"] = None
            p["down_seconds"] = 0.0
            p["last_lying_observation"] = None
            p["last_drop_time"] = None
            self._clear_motion_history(p)

        return self._result(
            p,
            smooth_angle,
            ratio,
            hip_speed,
            chest_speed,
            head_speed,
            chest_down_speed,
            head_down_speed,
            new_alert,
            torso_reliable,
            lying,
            aspect_support,
            shoulder_count,
            hip_count,
        )

    def _reset_possible(self, person):
        person["state"] = "normal"
        person["possible_since"] = None
        person["possible_observed_seconds"] = 0.0
        person["last_possible_observation"] = None
        person["down_seconds"] = 0.0
        person["last_lying_observation"] = None
        person["last_drop_time"] = None
        self._clear_motion_history(person)

    def _clear_motion_history(self, person):
        for key in ("hip_hist", "chest_hist", "head_hist"):
            person[key].clear()

    def _result(
        self,
        person,
        angle,
        ratio,
        hip_speed,
        chest_speed,
        head_speed,
        chest_down_speed,
        head_down_speed,
        new_alert,
        measurement_reliable,
        lying,
        aspect_support,
        shoulder_count,
        hip_count,
    ):
        return {
            "state": person["state"],
            "is_falling": person["state"] == "alerted",
            "new_alert": new_alert,
            "measurement_reliable": measurement_reliable,
            "lying_pose": lying,
            "aspect_support": aspect_support,
            "valid_shoulders": shoulder_count,
            "valid_hips": hip_count,
            "hip_speed": hip_speed,
            "chest_speed": chest_speed,
            "head_speed": head_speed,
            "chest_down_speed": chest_down_speed,
            "head_down_speed": head_down_speed,
            "telemetry": (
                f"{person['state']} | ang {_fmt(angle, 0)} "
                f"| ratio {_fmt(ratio)} | spd {_fmt(hip_speed)} "
                f"| chest {_fmt(chest_speed)} | head {_fmt(head_speed)}"
            ),
        }

    def forget_old(self, now, max_age=None):
        """Remove tracks after their configured or explicitly supplied timeout."""
        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(now)
        ):
            raise ValueError("now must be a finite timestamp")
        age_limit = self.cfg["track_timeout_sec"] if max_age is None else max_age
        if (
            isinstance(age_limit, bool)
            or not isinstance(age_limit, (int, float))
            or not math.isfinite(age_limit)
            or age_limit <= 0
        ):
            raise ValueError("max_age must be a positive finite number")
        old = [
            track_id
            for track_id, person in self.people.items()
            if now - person["last_seen"] > age_limit
        ]
        for track_id in old:
            del self.people[track_id]
