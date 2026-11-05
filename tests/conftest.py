import pytest

from src.events.fall import FallAnalyzer


@pytest.fixture
def fall_config():
    """A fresh copy of the fall settings used by the project."""
    return {
        "min_joint_conf": 0.5,
        "smooth_frames": 5,
        "angle_deg": 60,
        "recover_angle_deg": 40,
        "aspect_ratio": 1.2,
        "speed_window_sec": 0.5,
        "min_torso_length_px": 10,
        "min_speed_interval_sec": 0.05,
        "drop_speed": 1.5,
        "drop_window_sec": 2.0,
        "stay_down_sec": 3.0,
        "possible_timeout_sec": 4.0,
        "cooldown_sec": 30,
        "track_timeout_sec": 10,
    }


@pytest.fixture
def fall_analyzer(fall_config):
    return FallAnalyzer(fall_config)


@pytest.fixture
def confidence_values():
    return [1.0] * 17


@pytest.fixture
def make_keypoints():
    """Create simple COCO-shaped standing, sitting, or horizontal poses."""
    def make_pose(pose="standing", hip_y=70):
        offset = hip_y - 70

        if pose == "lying":
            y = hip_y
            return [
                [10, y], [15, y - 5], [15, y + 5], [20, y - 8], [20, y + 8],
                [30, y], [50, y], [25, y - 10], [55, y + 10],
                [0, y - 10], [70, y + 10], [70, y], [90, y],
                [80, y - 10], [80, y + 10], [100, y - 10], [100, y + 10],
            ]

        points = [
            [50, 15 + offset], [45, 10 + offset], [55, 10 + offset],
            [40, 15 + offset], [60, 15 + offset],
            [40, 30 + offset], [60, 30 + offset],
            [30, 45 + offset], [70, 45 + offset],
            [25, 55 + offset], [75, 55 + offset],
            [45, 70 + offset], [55, 70 + offset],
            [45, 85 + offset], [55, 85 + offset],
            [45, 100 + offset], [55, 100 + offset],
        ]

        if pose == "sitting":
            points[13] = [35, 85 + offset]
            points[14] = [65, 85 + offset]
            points[15] = [25, 85 + offset]
            points[16] = [75, 85 + offset]
        elif pose == "crawling":
            points[5] = [30, 45 + offset]
            points[6] = [50, 50 + offset]
            points[11] = [60, 65 + offset]
            points[12] = [80, 70 + offset]
            points[13] = [75, 75 + offset]
            points[14] = [90, 75 + offset]
            points[15] = [90, 85 + offset]
            points[16] = [100, 85 + offset]

        return points

    return make_pose


@pytest.fixture
def standing_keypoints(make_keypoints):
    return make_keypoints("standing")


@pytest.fixture
def lying_keypoints(make_keypoints):
    return make_keypoints("lying", hip_y=100)
