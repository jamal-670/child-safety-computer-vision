"""Main pipeline: camera -> pose + tracking -> fall analysis -> screen.
Run from the project root:  python -m src.main
"""
import time

import cv2
import yaml
from ultralytics import YOLO

from src.events.fall import FallAnalyzer
from src.utils.draw import draw_skeleton, put_text


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    model = YOLO(cfg["model"]["path"])
    analyzer = FallAnalyzer(cfg["fall"])
    cap = cv2.VideoCapture(cfg["camera"]["source"])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()

        # track() = detect + keep the same ID for the same person
        results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              conf=cfg["model"]["conf"], verbose=False)
        r = results[0]
        display = frame.copy()

        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().tolist()
            xy = r.keypoints.xy.cpu().numpy()
            conf = r.keypoints.conf
            conf = conf.cpu().numpy() if conf is not None else None

            for i, track_id in enumerate(ids):
                person_conf = conf[i] if conf is not None else None
                result = analyzer.analyze_person(track_id, xy[i], person_conf, now)

                draw_skeleton(display, xy[i], person_conf, cfg["fall"]["min_joint_conf"])
                put_text(display, f"ID {track_id}: {result['telemetry']}", (10, 25 + i * 20))

                if result["is_falling"]:
                    put_text(display, "ALERT: FALL DETECTED!", (50, 70), (0, 0, 255), 1.0, 3)
                if result["new_alert"]:
                    print(f"[ALERT] possible fall, person {track_id}, time {time.strftime('%H:%M:%S')}")

        analyzer.forget_old(now)

        cv2.imshow("Daycare Safety", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()