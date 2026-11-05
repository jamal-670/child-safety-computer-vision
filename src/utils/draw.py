
"""Drawing only. No fall decisions are made here."""

import cv2

from src.pose.constants import KEYPOINT_NAMES, SKELETON_CONNECTIONS
from src.utils.geometry import get_point


def draw_skeleton(frame, kpts, confs=None, min_conf=0.5, labels=False):
    points = [
        get_point(kpts, confs, i, min_conf)
        for i in range(17)
    ]

    for a, b in SKELETON_CONNECTIONS:
        if points[a] is not None and points[b] is not None:
            p1 = tuple(int(v) for v in points[a])
            p2 = tuple(int(v) for v in points[b])
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

    for i, point in enumerate(points):
        if point is None:
            continue

        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        if labels:
            label = f"{KEYPOINT_NAMES[i]} ({x}, {y})"
            cv2.putText(
                frame,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
            )


def put_text(
    frame,
    text,
    pos,
    color=(255, 255, 255),
    scale=0.5,
    thickness=1,
):
    cv2.putText(
        frame,
        str(text),
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
    )
