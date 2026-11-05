
"""Pure geometry helpers: no camera, model, or tracking state."""

import math


def get_point(kpts, confs, idx, min_conf=0.5):
    """Return a valid (x, y) point or None."""
    if kpts is None or idx < 0 or idx >= len(kpts):
        return None

    if confs is not None:
        if idx >= len(confs):
            return None

    try:
        x = float(kpts[idx][0])
        y = float(kpts[idx][1])

        if not math.isfinite(x) or not math.isfinite(y):
            return None

        if confs is not None:
            confidence = float(confs[idx])
            if not math.isfinite(confidence) or confidence < min_conf:
                return None

        return (x, y)

    except (TypeError, ValueError, IndexError):
        return None


def midpoint(p1, p2):
    """Return the midpoint; use the available point if only one exists."""
    points = [p for p in (p1, p2) if p is not None]

    if not points:
        return None

    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def distance(p1, p2):
    """Return distance between two points, or None if either is missing."""
    if p1 is None or p2 is None:
        return None

    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def torso_angle(shoulder_mid, hip_mid):
    """Angle from vertical: 0 degrees is upright, 90 degrees is horizontal."""
    if shoulder_mid is None or hip_mid is None:
        return None

    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]

    if math.hypot(dx, dy) < 1.0:
        return None

    return abs(math.degrees(math.atan2(dx, dy)))


def aspect_ratio(points):
    """Estimate width / height from at least five visible keypoints."""
    points = [p for p in points if p is not None]

    if len(points) < 5:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    return width / max(height, 1.0)
