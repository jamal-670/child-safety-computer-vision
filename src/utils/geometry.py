"""Pure maths helpers. No camera, no model, no memory.
A missing value is always None, never a fake number."""
import math


def get_point(kpts, confs, idx, min_conf):
    """Return (x, y) for one joint, or None if it is missing or unsure."""
    x, y = float(kpts[idx][0]), float(kpts[idx][1])
    if x <= 0 or y <= 0:
        return None
    if confs is not None and float(confs[idx]) < min_conf:
        return None
    return (x, y)


def midpoint(p1, p2):
    """Middle of two points. If one is missing, use the other. Both missing -> None."""
    pts = [p for p in (p1, p2) if p is not None]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts))


def distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def torso_angle(shoulder_mid, hip_mid):
    """Tilt of the torso from vertical, in degrees. 0 = upright, 90 = flat."""
    if shoulder_mid is None or hip_mid is None:
        return None
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]  # positive when upright, because y grows downward
    return abs(math.degrees(math.atan2(dx, dy)))


def aspect_ratio(points):
    """Width / height of the box around all visible joints. Needs at least 5 joints."""
    pts = [p for p in points if p is not None]
    if len(pts) < 5:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(xs) - min(xs)) / max(max(ys) - min(ys), 1.0)