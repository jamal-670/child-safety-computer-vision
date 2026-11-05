import pytest

from src.events.fall import FallAnalyzer


def _begin_possible_fall(analyzer, make_keypoints, confidences, track_id=1):
    """Create the standing-to-lying sequence needed for a possible fall."""
    analyzer.analyze_person(
        track_id, make_keypoints("standing", hip_y=70), confidences, 0.0
    )
    for frame in range(1, 5):
        result = analyzer.analyze_person(
            track_id,
            make_keypoints("lying", hip_y=100),
            confidences,
            frame / 10,
        )
        if result["state"] == "possible":
            return result
    return result


def _translate(points, dx=0, dy=0):
    return [[x + dx, y + dy] for x, y in points]


def test_standing_person_stays_normal(
    fall_analyzer, standing_keypoints, confidence_values
):
    result = fall_analyzer.analyze_person(
        1, standing_keypoints, confidence_values, 0.0
    )

    assert result["state"] == "normal"
    assert result["is_falling"] is False
    assert result["new_alert"] is False


@pytest.mark.parametrize("pose", ["sitting", "crawling"])
def test_non_fall_poses_do_not_trigger_fall(
    fall_analyzer, make_keypoints, confidence_values, pose
):
    result = fall_analyzer.analyze_person(
        1, make_keypoints(pose), confidence_values, 0.0
    )

    assert result["state"] == "normal"
    assert result["new_alert"] is False


def test_lying_without_preceding_drop_does_not_trigger_fall(
    fall_analyzer, lying_keypoints, confidence_values
):
    result = fall_analyzer.analyze_person(
        1, lying_keypoints, confidence_values, 0.0
    )

    assert result["state"] == "normal"
    assert result["new_alert"] is False


def test_fast_downward_movement_and_lying_pose_enter_possible_state(
    fall_analyzer, make_keypoints, confidence_values
):
    result = _begin_possible_fall(
        fall_analyzer, make_keypoints, confidence_values
    )

    assert result["state"] == "possible"
    assert result["hip_speed"] > fall_analyzer.cfg["drop_speed"]
    assert result["new_alert"] is False


def test_hip_speed_uses_representative_torso_length(fall_analyzer, make_keypoints,
                                                     confidence_values):
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)

    # Each stored hip sample retains its contemporaneous torso length.
    history = fall_analyzer.people[1]["hip_hist"]
    assert all(len(sample) == 4 for sample in history)


def test_head_and_chest_speeds_are_available_when_points_are_visible(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.analyze_person(
        1, make_keypoints("standing", hip_y=70), confidence_values, 0.0
    )
    result = fall_analyzer.analyze_person(
        1, make_keypoints("standing", hip_y=80), confidence_values, 0.1
    )

    assert result["chest_speed"] is not None
    assert result["head_speed"] is not None
    assert "chest " in result["telemetry"]
    assert "head " in result["telemetry"]


def test_repeated_timestamp_does_not_create_velocity(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.analyze_person(
        1, make_keypoints("standing", hip_y=70), confidence_values, 0.0
    )
    fall_analyzer.analyze_person(
        1, make_keypoints("standing", hip_y=70), confidence_values, 0.1
    )
    result = fall_analyzer.analyze_person(
        1, make_keypoints("lying", hip_y=100), confidence_values, 0.1
    )

    assert result["hip_speed"] is None


def test_coordinated_head_and_chest_drop_can_support_fall_evidence(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.analyze_person(
        1, make_keypoints("standing", hip_y=70), confidence_values, 0.0
    )
    # Upper body moves down into a horizontal pose while hips stay at the same y.
    for frame in range(1, 4):
        result = fall_analyzer.analyze_person(
            1,
            make_keypoints("lying", hip_y=70),
            confidence_values,
            frame / 10,
        )

    assert result["hip_speed"] == 0
    assert fall_analyzer.people[1]["last_drop_time"] is not None
    assert result["state"] == "possible"


def test_missing_nose_does_not_prevent_torso_detection(
    fall_analyzer, make_keypoints, confidence_values
):
    confidences = confidence_values.copy()
    confidences[0] = 0.0

    fall_analyzer.analyze_person(
        1,
        make_keypoints("standing", hip_y=70),
        confidence_values,
        0.0,
    )
    result = fall_analyzer.analyze_person(
        1, make_keypoints("standing", hip_y=80), confidences, 0.1
    )

    assert result["chest_speed"] is not None
    assert result["head_speed"] is None


def test_missing_nose_for_multiple_frames_does_not_disable_other_speeds(
    fall_analyzer, make_keypoints, confidence_values
):
    confidence_without_nose = confidence_values.copy()
    confidence_without_nose[0] = 0.0
    fall_analyzer.analyze_person(
        1, make_keypoints("standing", hip_y=70), confidence_values, 0.0
    )

    for frame in range(1, 9):
        result = fall_analyzer.analyze_person(
            1,
            make_keypoints("lying", hip_y=100),
            confidence_without_nose,
            frame / 10,
        )

    assert result["head_speed"] is None
    assert result["chest_speed"] is not None
    assert result["hip_speed"] is not None


def test_nose_return_after_long_gap_does_not_create_head_speed(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.cfg["max_observation_gap_sec"] = 0.2
    fall_analyzer.analyze_person(
        1, make_keypoints("standing", hip_y=70), confidence_values, 0.0
    )
    missing_nose = confidence_values.copy()
    missing_nose[0] = 0.0
    fall_analyzer.analyze_person(
        1, make_keypoints("standing", hip_y=80), confidence_values, 0.1
    )
    for frame in range(2, 6):
        fall_analyzer.analyze_person(
            1,
            make_keypoints("standing", hip_y=80 + frame),
            missing_nose,
            frame / 10,
        )

    result = fall_analyzer.analyze_person(
        1, make_keypoints("standing", hip_y=100), confidence_values, 0.6
    )

    assert result["head_speed"] is None
    assert len(fall_analyzer.people[1]["head_hist"]) == 1


def test_hip_based_fall_detection_works_without_nose(
    fall_analyzer, make_keypoints, confidence_values
):
    confidence_without_nose = confidence_values.copy()
    confidence_without_nose[0] = 0.0
    fall_analyzer.analyze_person(
        1, make_keypoints("standing", hip_y=70), confidence_without_nose, 0.0
    )
    for frame in range(1, 5):
        result = fall_analyzer.analyze_person(
            1,
            make_keypoints("lying", hip_y=100),
            confidence_without_nose,
            frame / 10,
        )

    assert result["hip_speed"] > fall_analyzer.cfg["drop_speed"]
    assert result["state"] == "possible"


def test_chest_based_drop_detection_works_without_nose(
    fall_analyzer, make_keypoints, confidence_values
):
    confidence_without_nose = confidence_values.copy()
    confidence_without_nose[0] = 0.0
    fall_analyzer.analyze_person(
        1, make_keypoints("standing", hip_y=70), confidence_without_nose, 0.0
    )
    for frame in range(1, 5):
        result = fall_analyzer.analyze_person(
            1,
            make_keypoints("lying", hip_y=70),
            confidence_without_nose,
            frame / 10,
        )

    assert result["hip_speed"] == 0
    assert result["chest_down_speed"] > fall_analyzer.cfg["drop_speed"]
    assert result["head_down_speed"] is None
    assert result["state"] == "possible"


@pytest.mark.parametrize("missing_index", [5, 11])
def test_one_missing_shoulder_or_hip_is_not_a_reliable_torso(
    fall_analyzer, make_keypoints, confidence_values, missing_index
):
    points = make_keypoints("lying", hip_y=100)
    points[missing_index] = None

    result = fall_analyzer.analyze_person(
        1, points, confidence_values, 0.0
    )

    assert result["state"] == "normal"
    assert result["measurement_reliable"] is False
    assert result["hip_speed"] is None
    assert result["valid_shoulders"] < 2 or result["valid_hips"] < 2


def test_low_confidence_torso_points_are_unreliable(
    fall_analyzer, standing_keypoints, confidence_values
):
    confidences = confidence_values.copy()
    confidences[5] = 0.1
    result = fall_analyzer.analyze_person(
        1, standing_keypoints, confidences, 0.0
    )

    assert result["state"] == "normal"
    assert result["hip_speed"] is None


def test_missing_aspect_ratio_does_not_reject_reliable_lying_angle(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.analyze_person(
        1, make_keypoints("standing", hip_y=70), confidence_values, 0.0
    )
    points = make_keypoints("lying", hip_y=100)
    for index in (13, 14, 15, 16):
        points[index] = None

    for frame in range(1, 5):
        result = fall_analyzer.analyze_person(
            1, points, confidence_values, frame / 10
        )

    assert result["measurement_reliable"] is True
    assert result["aspect_support"] is False
    assert result["lying_pose"] is True
    assert result["state"] == "possible"


def test_aspect_ratio_alone_cannot_confirm_lying_pose(
    fall_analyzer, make_keypoints, confidence_values
):
    points = make_keypoints("standing")
    points[13] = [0, 0]
    points[14] = [200, 0]
    points[15] = [0, 1]
    points[16] = [200, 1]

    result = fall_analyzer.analyze_person(
        1, points, confidence_values, 0.0
    )

    assert result["aspect_support"] is True
    assert result["lying_pose"] is False
    assert result["state"] == "normal"


def test_aspect_ratio_supports_a_borderline_torso_angle(
    fall_analyzer, make_keypoints, confidence_values
):
    points = make_keypoints("standing", hip_y=70)
    points[11] = [75, 55]
    points[12] = [85, 55]
    points[13] = [0, 70]
    points[14] = [150, 70]
    points[15] = [0, 100]
    points[16] = [150, 100]
    for frame in range(3):
        result = fall_analyzer.analyze_person(
            1, points, confidence_values, frame / 10
        )

    assert fall_analyzer.cfg["recover_angle_deg"] < 50
    assert result["aspect_support"] is True
    assert result["lying_pose"] is True


def test_upright_recovery_does_not_need_aspect_ratio(
    fall_analyzer, make_keypoints, confidence_values
):
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)
    points = make_keypoints("standing", hip_y=100)
    for index in (13, 14, 15, 16):
        points[index] = None

    for frame in range(5, 10):
        result = fall_analyzer.analyze_person(
            1, points, confidence_values, frame / 10
        )

    assert result["aspect_support"] is False
    assert result["state"] == "normal"


def test_upright_recovery_is_not_blocked_by_spread_arms(
    fall_analyzer, make_keypoints, confidence_values
):
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)
    points = make_keypoints("standing", hip_y=100)
    points[7] = [-100, 40]
    points[9] = [-150, 40]
    points[8] = [200, 40]
    points[10] = [250, 40]

    for frame in range(5, 10):
        result = fall_analyzer.analyze_person(
            1, points, confidence_values, frame / 10
        )

    assert result["state"] == "normal"


def test_invalid_coordinates_are_ignored_safely(
    fall_analyzer, standing_keypoints, confidence_values
):
    points = [point.copy() for point in standing_keypoints]
    points[0] = [float("nan"), 10]
    result = fall_analyzer.analyze_person(1, points, confidence_values, 0.0)

    assert result["state"] == "normal"
    assert result["head_speed"] is None


def test_person_remaining_down_confirms_alert_after_configured_time(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.cfg["stay_down_sec"] = 0.2
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)

    result = fall_analyzer.analyze_person(
        1, make_keypoints("lying", hip_y=100), confidence_values, 0.6
    )

    assert result["state"] == "alerted"
    assert result["is_falling"] is True
    assert result["new_alert"] is True


def test_alert_is_emitted_once_while_person_remains_alerted(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.cfg["stay_down_sec"] = 0.2
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)
    first = fall_analyzer.analyze_person(
        1, make_keypoints("lying", hip_y=100), confidence_values, 0.6
    )
    second = fall_analyzer.analyze_person(
        1, make_keypoints("lying", hip_y=100), confidence_values, 0.7
    )

    assert first["new_alert"] is True
    assert second["state"] == "alerted"
    assert second["new_alert"] is False


def test_recovery_before_confirmation_cancels_possible_fall(
    fall_analyzer, make_keypoints, confidence_values
):
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)

    for frame in range(5, 9):
        result = fall_analyzer.analyze_person(
            1,
            make_keypoints("standing", hip_y=100),
            confidence_values,
            frame / 10,
        )

    assert result["state"] == "normal"
    assert result["new_alert"] is False
    assert fall_analyzer.people[1]["down_seconds"] == 0


def test_nonlying_pose_does_not_advance_stay_down_timer(
    fall_analyzer, make_keypoints, confidence_values
):
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)

    for frame in range(5, 11):
        result = fall_analyzer.analyze_person(
            1,
            make_keypoints("standing", hip_y=100),
            confidence_values,
            frame / 10,
        )

    assert result["state"] == "normal"
    assert fall_analyzer.people[1]["down_seconds"] == 0


def test_moving_while_lying_does_not_cancel_possible_fall(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.cfg["stay_down_sec"] = 0.7
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)

    first = fall_analyzer.analyze_person(
        1,
        _translate(make_keypoints("lying", hip_y=100), dx=5),
        confidence_values,
        0.5,
    )
    second = fall_analyzer.analyze_person(
        1,
        _translate(make_keypoints("lying", hip_y=100), dx=10),
        confidence_values,
        0.6,
    )
    third = fall_analyzer.analyze_person(
        1,
        _translate(make_keypoints("lying", hip_y=100), dx=15),
        confidence_values,
        0.7,
    )
    fall_analyzer.analyze_person(
        1,
        _translate(make_keypoints("lying", hip_y=100), dx=20),
        confidence_values,
        0.8,
    )
    final = fall_analyzer.analyze_person(
        1,
        _translate(make_keypoints("lying", hip_y=100), dx=25),
        confidence_values,
        0.9,
    )

    assert first["state"] == "possible"
    assert second["state"] == "possible"
    assert third["state"] == "possible"
    assert final["state"] == "alerted"


def test_missing_frame_does_not_reset_short_possible_gap(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.cfg["stay_down_sec"] = 0.2
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)
    missing = fall_analyzer.analyze_person(1, None, None, 0.5)
    resumed = fall_analyzer.analyze_person(
        1, make_keypoints("lying", hip_y=100), confidence_values, 0.6
    )

    assert missing["state"] == "possible"
    assert resumed["state"] == "alerted"


def test_intermittent_short_missing_observation_can_confirm_fall(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.cfg["stay_down_sec"] = 0.3
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)
    fall_analyzer.analyze_person(1, None, None, 0.5)
    resumed = fall_analyzer.analyze_person(
        1, make_keypoints("lying", hip_y=100), confidence_values, 0.6
    )

    assert resumed["state"] == "alerted"


def test_long_observation_gap_does_not_count_as_staying_down(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.cfg["stay_down_sec"] = 0.2
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)

    result = fall_analyzer.analyze_person(
        1, make_keypoints("lying", hip_y=100), confidence_values, 2.0
    )

    assert result["state"] == "possible"
    assert result["new_alert"] is False
    assert fall_analyzer.people[1]["down_seconds"] == 0


def test_long_missing_gap_does_not_advance_possible_timeout(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.cfg["possible_timeout_sec"] = 1.0
    fall_analyzer.cfg["stay_down_sec"] = 0.4
    fall_analyzer.cfg["max_observation_gap_sec"] = 0.2
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)
    observed_before_gap = fall_analyzer.people[1]["possible_observed_seconds"]

    missing = fall_analyzer.analyze_person(1, None, None, 5.0)
    assert missing["state"] == "possible"
    assert (
        fall_analyzer.people[1]["possible_observed_seconds"]
        == observed_before_gap
    )

    for frame in range(1, 9):
        result = fall_analyzer.analyze_person(
            1,
            make_keypoints("lying", hip_y=100),
            confidence_values,
            5.0 + frame / 10,
        )

    assert result["state"] == "alerted"


def test_valid_fall_is_not_cancelled_by_one_long_observation_gap(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.cfg["possible_timeout_sec"] = 0.7
    fall_analyzer.cfg["stay_down_sec"] = 0.2
    fall_analyzer.cfg["max_observation_gap_sec"] = 0.2
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)
    fall_analyzer.analyze_person(1, None, None, 5.0)

    for frame in range(1, 8):
        result = fall_analyzer.analyze_person(
            1,
            make_keypoints("lying", hip_y=100),
            confidence_values,
            5.0 + frame / 10,
        )

    assert result["state"] == "alerted"


def test_possible_fall_expires_after_configured_timeout(
    fall_config, make_keypoints, confidence_values
):
    fall_config["possible_timeout_sec"] = 0.5
    fall_config["stay_down_sec"] = 0.4
    fall_analyzer = FallAnalyzer(fall_config)
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)

    # A torso angle between recovery and lying keeps the event unresolved.
    intermediate_pose = make_keypoints("standing", hip_y=70)
    intermediate_pose[11] = [75, 55]
    intermediate_pose[12] = [85, 55]
    for frame in range(5, 11):
        result = fall_analyzer.analyze_person(
            1, intermediate_pose, confidence_values, frame / 10
        )

    assert result["state"] == "normal"
    assert result["new_alert"] is False
    assert fall_analyzer.people[1]["last_drop_time"] is None


def test_tracks_have_independent_detection_state(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.analyze_person(
        1, make_keypoints("standing"), confidence_values, 0.0
    )
    fall_analyzer.analyze_person(
        2, make_keypoints("standing"), confidence_values, 0.0
    )
    for frame in range(1, 5):
        first = fall_analyzer.analyze_person(
            1, make_keypoints("lying", hip_y=100), confidence_values, frame / 10
        )
        second = fall_analyzer.analyze_person(
            2, make_keypoints("standing"), confidence_values, frame / 10
        )

    assert first["state"] == "possible"
    assert second["state"] == "normal"
    assert fall_analyzer.people[1]["state"] != fall_analyzer.people[2]["state"]


def test_alert_cooldown_suppresses_a_second_alert(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.cfg["stay_down_sec"] = 0.2
    fall_analyzer.cfg["cooldown_sec"] = 30
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)
    first_alert = fall_analyzer.analyze_person(
        1, make_keypoints("lying", hip_y=100), confidence_values, 0.6
    )

    for frame in range(7, 12):
        fall_analyzer.analyze_person(
            1, make_keypoints("standing", hip_y=100), confidence_values, frame / 10
        )
    for frame in range(12, 16):
        result = fall_analyzer.analyze_person(
            1, make_keypoints("lying", hip_y=140), confidence_values, frame / 10
        )
    second_alert = fall_analyzer.analyze_person(
        1, make_keypoints("lying", hip_y=140), confidence_values, 1.8
    )

    assert first_alert["new_alert"] is True
    assert result["state"] == "possible"
    assert second_alert["state"] == "alerted"
    assert second_alert["new_alert"] is False


def test_movement_while_alerted_does_not_clear_alert(
    fall_analyzer, make_keypoints, confidence_values
):
    fall_analyzer.cfg["stay_down_sec"] = 0.2
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)
    fall_analyzer.analyze_person(
        1, make_keypoints("lying", hip_y=100), confidence_values, 0.6
    )

    result = fall_analyzer.analyze_person(
        1,
        _translate(make_keypoints("lying", hip_y=100), dx=15),
        confidence_values,
        0.7,
    )

    assert result["state"] == "alerted"
    assert result["new_alert"] is False


def test_missing_keypoints_preserve_possible_state(
    fall_analyzer, make_keypoints, confidence_values
):
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)

    result = fall_analyzer.analyze_person(1, None, None, 0.5)

    assert result["state"] == "possible"
    assert result["is_falling"] is False
    assert result["hip_speed"] is None


def test_forget_old_uses_configured_tracking_timeout(
    fall_analyzer, standing_keypoints, confidence_values
):
    fall_analyzer.analyze_person(7, standing_keypoints, confidence_values, 1.0)

    fall_analyzer.forget_old(now=11.1)

    assert 7 not in fall_analyzer.people


def test_reused_track_id_after_timeout_gets_fresh_fall_memory(
    fall_analyzer, make_keypoints, confidence_values
):
    _begin_possible_fall(fall_analyzer, make_keypoints, confidence_values)

    result = fall_analyzer.analyze_person(
        1, make_keypoints("lying", hip_y=100), confidence_values, 11.0
    )

    assert result["state"] == "normal"
    assert fall_analyzer.people[1]["last_drop_time"] is None


def test_non_finite_timestamp_is_rejected(
    fall_analyzer, standing_keypoints, confidence_values
):
    with pytest.raises(ValueError, match="finite timestamp"):
        fall_analyzer.analyze_person(
            1, standing_keypoints, confidence_values, float("nan")
        )


def test_invalid_configuration_is_rejected(fall_config):
    fall_config["drop_speed"] = float("nan")

    with pytest.raises(ValueError, match="drop_speed"):
        FallAnalyzer(fall_config)
