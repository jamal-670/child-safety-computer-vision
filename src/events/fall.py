"""Fall detection from skeleton points.

Idea: a fall = a FAST DROP, then a LYING pose, then STAYING DOWN for a few seconds.
States per person: normal -> possible -> alerted.
"""
from collections import deque

from src.pose.constants import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER
from src.utils import geometry as geo


def _fmt(value, digits=1):
    return "--" if value is None else f"{value:.{digits}f}"


class FallAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.people = {}  # track_id -> memory for that person

    def _new_memory(self):
        return {
            "hip_hist": deque(),                                  # (time, hip_y)
            "angles": deque(maxlen=self.cfg["smooth_frames"]),    # recent angles
            "state": "normal",
            "possible_since": None,
            "last_drop_time": None,
            "last_alert_time": None,
            "last_seen": 0.0,
        }

    def analyze_person(self, track_id, kpts, confs, now):
        """kpts: 17 (x, y) points. confs: 17 confidences or None. now: time in seconds."""
        cfg = self.cfg
        p = self.people.setdefault(track_id, self._new_memory())
        p["last_seen"] = now

        # 1. Read joints. A missing joint is None.
        pts = [geo.get_point(kpts, confs, i, cfg["min_joint_conf"]) for i in range(17)]
        shoulder_mid = geo.midpoint(pts[L_SHOULDER], pts[R_SHOULDER])
        hip_mid = geo.midpoint(pts[L_HIP], pts[R_HIP])

        angle = geo.torso_angle(shoulder_mid, hip_mid)
        torso_len = geo.distance(shoulder_mid, hip_mid)
        ratio = geo.aspect_ratio(pts)

        # 2. If we cannot measure the torso, say "unknown" and keep the old state.
        if angle is None or torso_len is None or torso_len < 10:
            return self._result(p, None, ratio, None, False)

        # 3. Smooth the angle over the last few frames.
        p["angles"].append(angle)
        smooth_angle = sum(p["angles"]) / len(p["angles"])

        # 4. Hip drop speed in torso-lengths per second (same at any distance).
        hist = p["hip_hist"]
        hist.append((now, hip_mid[1]))
        while hist and now - hist[0][0] > cfg["speed_window_sec"]:
            hist.popleft()
        speed = None
        if len(hist) >= 2:
            dt = now - hist[0][0]
            if dt > 0.05:
                speed = (hip_mid[1] - hist[0][1]) / dt / torso_len  # positive = going down

        if speed is not None and speed > cfg["drop_speed"]:
            p["last_drop_time"] = now
        recent_drop = (p["last_drop_time"] is not None
                       and now - p["last_drop_time"] <= cfg["drop_window_sec"])

        # 5. Pose checks.
        lying = smooth_angle > cfg["angle_deg"] and (ratio is None or ratio > cfg["aspect_ratio"])
        recovered = smooth_angle < cfg["recover_angle_deg"]

        # 6. State machine.
        new_alert = False
        if p["state"] == "normal":
            if lying and recent_drop:
                p["state"] = "possible"
                p["possible_since"] = now
        elif p["state"] == "possible":
            if recovered:
                p["state"] = "normal"
            elif now - p["possible_since"] >= cfg["stay_down_sec"]:
                p["state"] = "alerted"
                last = p["last_alert_time"]
                if last is None or now - last >= cfg["cooldown_sec"]:
                    new_alert = True
                    p["last_alert_time"] = now
        elif p["state"] == "alerted":
            if recovered:
                p["state"] = "normal"

        return self._result(p, smooth_angle, ratio, speed, new_alert)

    def _result(self, p, angle, ratio, speed, new_alert):
        return {
            "state": p["state"],
            "is_falling": p["state"] == "alerted",
            "new_alert": new_alert,  # True only on the single frame the alert starts
            "telemetry": f"{p['state']} | ang {_fmt(angle, 0)} | ratio {_fmt(ratio)} | spd {_fmt(speed)}",
        }

    def forget_old(self, now, max_age=5.0):
        """Delete memory of people not seen for max_age seconds."""
        old = [tid for tid, p in self.people.items() if now - p["last_seen"] > max_age]
        for tid in old:
            del self.people[tid]