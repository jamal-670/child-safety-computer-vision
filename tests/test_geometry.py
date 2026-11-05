import math

import pytest

from src.utils.geometry import (
    aspect_ratio,
    distance,
    get_point,
    midpoint,
    torso_angle,
)


def test_midpoint_calculates_middle_of_two_points():
    assert midpoint((2, 4), (6, 8)) == (4, 6)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ((2, 3), None, (2, 3)),
        (None, (4, 5), (4, 5)),
        (None, None, None),
    ],
)
def test_midpoint_handles_missing_points(first, second, expected):
    assert midpoint(first, second) == expected


def test_distance_calculates_euclidean_distance():
    assert distance((0, 0), (3, 4)) == 5


@pytest.mark.parametrize(("first", "second"), [(None, (1, 2)), ((1, 2), None)])
def test_distance_returns_none_when_a_point_is_missing(first, second):
    assert distance(first, second) is None


def test_get_point_returns_coordinates_with_sufficient_confidence():
    assert get_point([[12, 24]], [0.8], 0, min_conf=0.5) == (12.0, 24.0)


@pytest.mark.parametrize(
    ("keypoints", "confidences", "index"),
    [
        (None, None, 0),
        ([[1, 2]], None, -1),
        ([[1, 2]], None, 1),
        ([[1, 2]], [], 0),
    ],
)
def test_get_point_returns_none_for_missing_or_out_of_range_data(
    keypoints, confidences, index
):
    assert get_point(keypoints, confidences, index) is None


@pytest.mark.parametrize(
    ("confidence", "minimum"),
    [(0.49, 0.5), (float("nan"), 0.5), (float("inf"), 0.5)],
)
def test_get_point_rejects_low_or_invalid_confidence(confidence, minimum):
    assert get_point([[1, 2]], [confidence], 0, min_conf=minimum) is None


def test_get_point_accepts_confidence_at_the_threshold():
    assert get_point([[1, 2]], [0.5], 0, min_conf=0.5) == (1.0, 2.0)


@pytest.mark.parametrize(
    "point",
    [[float("nan"), 1], [1, float("inf")], [None, 1], [1]],
)
def test_get_point_rejects_invalid_coordinates(point):
    assert get_point([point], None, 0) is None


def test_get_point_allows_zero_and_negative_coordinates():
    assert get_point([[0, -3]], None, 0) == (0.0, -3.0)


def test_torso_angle_is_zero_for_an_upright_torso():
    assert torso_angle((5, 0), (5, 10)) == 0


def test_torso_angle_is_ninety_degrees_for_a_horizontal_torso():
    assert torso_angle((0, 2), (10, 2)) == 90


@pytest.mark.parametrize(
    ("shoulder", "hip"),
    [(None, (1, 2)), ((1, 2), None), ((0, 0), (0.5, 0.5))],
)
def test_torso_angle_returns_none_when_it_cannot_be_measured(shoulder, hip):
    assert torso_angle(shoulder, hip) is None


def test_aspect_ratio_is_width_divided_by_height():
    assert aspect_ratio([(0, 0), (4, 0), (0, 2), (4, 2), (2, 1)]) == 2


def test_aspect_ratio_returns_none_with_fewer_than_five_points():
    assert aspect_ratio([(0, 0)] * 4) is None


def test_aspect_ratio_uses_minimum_height_of_one():
    assert aspect_ratio([(0, 2)] * 5) == 0
