"""Drawing only. No decisions are made here."""
import cv2

from src.pose.constants import KEYPOINT_NAMES, SKELETON_CONNECTIONS
from src.utils.geometry import get_point


def draw_skeleton(frame, kpts, confs=None, min_conf=0.5, labels=False):
    pts = [get_point(kpts, confs, i, min_conf) for i in range(17)]

    for a, b in SKELETON_CONNECTIONS:
        if pts[a] is not None and pts[b] is not None:
            p1 = (int(pts[a][0]), int(pts[a][1]))
            p2 = (int(pts[b][0]), int(pts[b][1]))
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

    for i, p in enumerate(pts):
        if p is None:
            continue
        x, y = int(p[0]), int(p[1])
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        if labels:
            cv2.putText(frame, KEYPOINT_NAMES[i], (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


def put_text(frame, text, pos, color=(255, 255, 255), scale=0.5, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)