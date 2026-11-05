
"""SafeGuard main pipeline.

Supports:
- Webcam
- One video file
- A folder containing multiple videos

Run from the project root:
    python -m src.main
    python -m src.main --source data/raw/fall_test.mp4
    python -m src.main --source data/raw
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from src.events.fall import FallAnalyzer
from src.tracking.tracker import ByteTrackAdapter
from src.utils.draw import draw_skeleton, put_text


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def load_config(path="configs/config.yaml"):
    """Load project configuration."""
    with open(path, "r") as file:
        return yaml.safe_load(file)


def get_video_sources(source):
    """Return one or more video paths, or a webcam source."""

    if isinstance(source, int):
        return [source]

    path = Path(source)

    if path.is_dir():
        videos = sorted(
            file
            for file in path.iterdir()
            if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS
        )

        if not videos:
            raise FileNotFoundError(
                f"No supported videos found in folder: {path}"
            )

        return videos

    if not path.is_file():
        raise FileNotFoundError(f"Video source not found: {path}")

    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension '{path.suffix}': {path}"
        )

    return [path]


def video_frame_time(cap, frame_index, fps, previous_time=None):
    """Get media time from OpenCV, with a monotonic FPS fallback.

    Position timestamps are preferable for variable-frame-rate videos. Some
    codecs do not expose them reliably, so frame-index/FPS is the fallback.
    """
    position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    if isinstance(position_ms, (int, float)) and math.isfinite(position_ms):
        media_time = position_ms / 1000
        if media_time >= 0 and (
            previous_time is None or media_time > previous_time
        ):
            return media_time

    fallback = frame_index / fps
    if previous_time is not None and fallback <= previous_time:
        fallback = previous_time + 1 / fps
    return fallback


def analyze_tracked_people(people, analyzer, now, frame_index):
    """Pass each pose to fall analysis with a stable or frame-local ID.

    Frame-local IDs prevent unrelated detections without ByteTrack IDs from
    sharing state, but deliberately do not persist across frames.
    """
    analyzed = []
    for index, person in enumerate(people):
        track_id = person.track_id
        analysis_id = (
            track_id
            if track_id is not None
            else ("untracked", frame_index, index)
        )
        detection = analyzer.analyze_person(
            analysis_id,
            person.keypoints,
            person.confidence,
            now,
        )
        analyzed.append((person, detection))
    return analyzed


def process_source(source, cfg):
    """Process one webcam or video source."""
    cap = None
    try:
        cap = cv2.VideoCapture(
            str(source) if isinstance(source, Path) else source
        )

        if not cap.isOpened():
            print(f"[ERROR] Could not open source: {source}")
            return

        is_video_file = isinstance(source, Path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) if is_video_file else None
        if is_video_file and (
            not isinstance(video_fps, (int, float))
            or not math.isfinite(video_fps)
            or video_fps <= 0
        ):
            raise ValueError(f"Could not read FPS from video: {source}")

        # A fresh model, tracker, and analyzer prevent state crossing between files.
        model = YOLO(cfg["model"]["path"])
        tracker = ByteTrackAdapter(model, cfg["model"]["conf"])
        analyzer = FallAnalyzer(cfg["fall"])

        debug = cfg.get("debug", False)
        show_labels = cfg.get("display", {}).get("show_labels", True)
        telemetry_times = {}
        frame_index = 0
        previous_video_time = None
        fallback_warning_shown = False

        print(f"\n[START] Processing: {source}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or getattr(frame, "size", 1) == 0:
                print(f"[WARNING] Empty frame received from: {source}")
                break

            if is_video_file:
                now = video_frame_time(
                    cap, frame_index, video_fps, previous_video_time
                )
                previous_video_time = now
            else:
                now = time.monotonic()
            frame_index += 1

            people = tracker.track(frame)
            display = frame.copy()
            person_count = len(people)

            if debug:
                print(
                    f"[DEBUG] People: {person_count} | "
                    f"Tracker IDs: {[person.track_id for person in people]}"
                )

            if person_count:
                if (
                    any(person.track_id is None for person in people)
                    and not fallback_warning_shown
                ):
                    print(
                        "[WARNING] ByteTrack IDs unavailable. "
                        "Untracked detections use frame-local state and "
                        "will not be linked across frames."
                    )
                    fallback_warning_shown = True

                analyzed_people = analyze_tracked_people(
                    people, analyzer, now, frame_index
                )
                for i, (person, detection) in enumerate(analyzed_people):
                    track_id = person.track_id
                    display_id = track_id if track_id is not None else "untracked"

                    draw_skeleton(
                        display,
                        person.keypoints,
                        person.confidence,
                        cfg["fall"]["min_joint_conf"],
                        labels=show_labels
                    )

                    telemetry_key = (
                        track_id
                        if track_id is not None
                        else ("untracked", frame_index, i)
                    )
                    last_print = telemetry_times.get(telemetry_key, 0.0)

                    if now - last_print >= 1.0:
                        print(
                            f"ID {display_id} | "
                            f"{detection['telemetry']}"
                        )
                        telemetry_times[telemetry_key] = now

                    put_text(
                        display,
                        f"ID {display_id}: {detection['telemetry']}",
                        (10, 25 + i * 20)
                    )

                    if detection["is_falling"]:
                        put_text(
                            display,
                            "ALERT: FALL DETECTED!",
                            (50, 70),
                            (0, 0, 255),
                            1.0,
                            3
                        )

                    if detection["new_alert"]:
                        print(
                            f"[ALERT] Possible fall detected "
                            f"for person {display_id}"
                        )

            analyzer.forget_old(now)

            # Remove telemetry for people no longer tracked.
            for track_id in list(telemetry_times):
                if track_id not in analyzer.people:
                    del telemetry_times[track_id]

            put_text(
                display,
                f"SAFEGUARD | Source: {Path(source).name if is_video_file else 'Webcam'}",
                (10, display.shape[0] - 12),
                (0, 255, 255),
                0.5,
                1
            )

            cv2.imshow("SafeGuard - Fall Detection", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[STOP] Video processing stopped by user.")
                break

        print(f"[FINISHED] {source}")
    finally:
        try:
            if cap is not None:
                cap.release()
        finally:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="SafeGuard video and webcam processing"
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video file or folder containing videos"
    )

    args = parser.parse_args()

    cfg = load_config()

    source = args.source if args.source is not None else cfg["camera"]["source"]

    sources = get_video_sources(source)

    print(f"[INFO] Found {len(sources)} source(s).")

    for video_source in sources:
        process_source(video_source, cfg)


if __name__ == "__main__":
    main()
