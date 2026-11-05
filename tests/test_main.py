from pathlib import Path

import numpy as np
import pytest

import src.main as main_module
from src.main import (
    analyze_tracked_people,
    get_video_sources,
    load_config,
    video_frame_time,
)
from src.tracking.tracker import TrackedPerson


def test_load_config_reads_yaml_configuration(tmp_path):
    config_path = tmp_path / "test-config.yaml"
    config_path.write_text("camera:\n  source: 3\nfall:\n  stay_down_sec: 2\n")

    config = load_config(config_path)

    assert config["camera"]["source"] == 3
    assert config["fall"]["stay_down_sec"] == 2


def test_load_config_reads_project_configuration():
    config = load_config("configs/config.yaml")

    assert config["camera"]["source"] == "data/raw"
    assert config["fall"]["stay_down_sec"] > 0
    assert config["fall"]["max_observation_gap_sec"] > 0


def test_get_video_sources_returns_one_video_file(tmp_path):
    video = tmp_path / "sample.mp4"
    video.touch()

    assert get_video_sources(video) == [video]


def test_get_video_sources_returns_sorted_supported_videos_from_folder(tmp_path):
    later_video = tmp_path / "z-video.MKV"
    earlier_video = tmp_path / "a-video.mp4"
    unsupported_file = tmp_path / "notes.txt"
    later_video.touch()
    earlier_video.touch()
    unsupported_file.touch()

    assert get_video_sources(tmp_path) == [earlier_video, later_video]


def test_get_video_sources_ignores_directories_with_video_suffix(tmp_path):
    (tmp_path / "folder.mp4").mkdir()

    with pytest.raises(FileNotFoundError, match="No supported videos"):
        get_video_sources(tmp_path)


def test_get_video_sources_raises_for_empty_or_unsupported_video_folder(tmp_path):
    (tmp_path / "notes.txt").touch()

    with pytest.raises(FileNotFoundError, match="No supported videos"):
        get_video_sources(tmp_path)


def test_get_video_sources_raises_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="Video source not found"):
        get_video_sources(tmp_path / "missing.mp4")


def test_get_video_sources_rejects_unsupported_single_file_extension(tmp_path):
    unsupported = tmp_path / "clip.txt"
    unsupported.touch()

    with pytest.raises(ValueError, match="Unsupported video extension"):
        get_video_sources(unsupported)


def test_get_video_sources_returns_webcam_integer_without_opening_camera():
    assert get_video_sources(0) == [0]


class RecordingAnalyzer:
    def __init__(self):
        self.calls = []

    def analyze_person(self, track_id, keypoints, confidence, now):
        self.calls.append((track_id, keypoints, confidence, now))
        return {"state": "normal"}


def test_main_passes_tracker_ids_pose_confidence_and_time_to_analyzer():
    first_points = np.ones((17, 2))
    second_points = np.full((17, 2), 2)
    first_confidence = np.ones(17)
    people = [
        TrackedPerson(20, first_points, first_confidence),
        TrackedPerson(21, second_points, None),
    ]
    analyzer = RecordingAnalyzer()

    results = analyze_tracked_people(people, analyzer, now=2.5, frame_index=8)

    assert [call[0] for call in analyzer.calls] == [20, 21]
    assert analyzer.calls[0][1] is first_points
    assert analyzer.calls[0][2] is first_confidence
    assert analyzer.calls[1][1] is second_points
    assert analyzer.calls[1][2] is None
    assert all(call[3] == 2.5 for call in analyzer.calls)
    assert len(results) == 2


def test_missing_tracker_ids_receive_distinct_frame_local_fall_states():
    people = [
        TrackedPerson(None, np.zeros((17, 2)), None),
        TrackedPerson(None, np.ones((17, 2)), None),
    ]
    analyzer = RecordingAnalyzer()

    analyze_tracked_people(people, analyzer, now=1, frame_index=10)
    analyze_tracked_people(people, analyzer, now=2, frame_index=11)

    ids = [call[0] for call in analyzer.calls]
    assert ids == [
        ("untracked", 10, 0),
        ("untracked", 10, 1),
        ("untracked", 11, 0),
        ("untracked", 11, 1),
    ]
    assert len(set(ids)) == 4


class FakeCapture:
    def __init__(self, source, fps=30, frames=None, is_open=True):
        self.source = source
        self.fps = fps
        self.frames = list(frames or [])
        self.open = is_open
        self.released = False

    def isOpened(self):
        return self.open

    def get(self, prop):
        if prop == main_module.cv2.CAP_PROP_FPS:
            return self.fps
        if prop == main_module.cv2.CAP_PROP_POS_MSEC:
            return 0
        return 0

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        self.released = True


def _mock_pipeline(monkeypatch, captures):
    created_analyzers = []
    created_models = []

    class FakeAnalyzer:
        def __init__(self, config):
            self.people = {}
            self.config = config
            created_analyzers.append(self)

        def forget_old(self, now):
            pass

    class FakeTracker:
        def __init__(self, model, confidence):
            self.model = model
            self.confidence = confidence

        def track(self, frame):
            return []

    def fake_yolo(path):
        model = object()
        created_models.append((path, model))
        return model

    iterator = iter(captures)
    monkeypatch.setattr(main_module.cv2, "VideoCapture", lambda source: next(iterator))
    monkeypatch.setattr(main_module.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(main_module.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(main_module.cv2, "waitKey", lambda *args: 0)
    monkeypatch.setattr(main_module, "YOLO", fake_yolo)
    monkeypatch.setattr(main_module, "ByteTrackAdapter", FakeTracker)
    monkeypatch.setattr(main_module, "FallAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(main_module, "put_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "draw_skeleton", lambda *args, **kwargs: None)
    return created_analyzers, created_models


def test_process_source_releases_capture_when_video_cannot_open(
    monkeypatch, tmp_path, capsys
):
    capture = FakeCapture(tmp_path / "unopenable.mp4", is_open=False)
    _mock_pipeline(monkeypatch, [capture])

    main_module.process_source(tmp_path / "unopenable.mp4", load_config())

    assert capture.released is True
    assert "Could not open source" in capsys.readouterr().out


def test_process_source_headless_skips_window_calls(monkeypatch):
    capture = FakeCapture(0)
    _mock_pipeline(monkeypatch, [capture])
    monkeypatch.setattr(
        main_module.cv2,
        "imshow",
        lambda *args: pytest.fail("headless processing must not open a window"),
    )
    monkeypatch.setattr(
        main_module.cv2,
        "waitKey",
        lambda *args: pytest.fail("headless processing must not wait for input"),
    )
    monkeypatch.setattr(
        main_module.cv2,
        "destroyAllWindows",
        lambda: pytest.fail("headless processing must not manage windows"),
    )

    main_module.process_source(0, load_config(), headless=True)

    assert capture.released is True


def test_process_source_stops_on_empty_frame_and_releases_capture(
    monkeypatch, capsys
):
    capture = FakeCapture(
        0,
        frames=[np.empty((0, 0, 3))],
    )
    _mock_pipeline(monkeypatch, [capture])

    main_module.process_source(0, load_config())

    assert capture.released is True
    assert "Empty frame" in capsys.readouterr().out


def test_process_source_releases_capture_when_fps_is_invalid(monkeypatch, tmp_path):
    capture = FakeCapture(tmp_path / "bad-fps.mp4", fps=0)
    _mock_pipeline(monkeypatch, [capture])

    with pytest.raises(ValueError, match="Could not read FPS"):
        main_module.process_source(tmp_path / "bad-fps.mp4", load_config())

    assert capture.released is True


def test_process_source_releases_capture_if_tracker_raises(
    monkeypatch, tmp_path
):
    capture = FakeCapture(
        tmp_path / "tracker-error.mp4",
        frames=[np.zeros((16, 16, 3))],
    )
    _mock_pipeline(monkeypatch, [capture])

    class BrokenTracker:
        def __init__(self, model, confidence):
            pass

        def track(self, frame):
            raise RuntimeError("tracker failed")

    monkeypatch.setattr(main_module, "ByteTrackAdapter", BrokenTracker)

    with pytest.raises(RuntimeError, match="tracker failed"):
        main_module.process_source(
            tmp_path / "tracker-error.mp4", load_config()
        )

    assert capture.released is True


def test_separate_video_processing_creates_fresh_tracker_and_analyzer(
    monkeypatch, tmp_path
):
    frames = [np.zeros((16, 16, 3))]
    sources = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    captures = [
        FakeCapture(source, frames=frames.copy())
        for source in sources
    ]
    analyzers, models = _mock_pipeline(monkeypatch, captures)

    for source in sources:
        main_module.process_source(source, load_config())

    assert len(analyzers) == 2
    assert analyzers[0] is not analyzers[1]
    assert len(models) == 2
    assert models[0][1] is not models[1][1]
    assert all(capture.released for capture in captures)


@pytest.mark.parametrize(
    ("position_ms", "frame_index", "previous", "expected"),
    [
        (250, 10, 0.1, 0.25),
        (0, 10, 0.2, 10 / 30),
        (float("nan"), 6, None, 0.2),
    ],
)
def test_video_frame_time_uses_media_time_or_monotonic_fallback(
    monkeypatch, position_ms, frame_index, previous, expected
):
    class PositionCapture:
        def get(self, prop):
            assert prop == main_module.cv2.CAP_PROP_POS_MSEC
            return position_ms

    actual = video_frame_time(PositionCapture(), frame_index, 30, previous)

    assert actual == pytest.approx(expected)
