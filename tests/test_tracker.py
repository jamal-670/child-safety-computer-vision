from types import SimpleNamespace

import numpy as np
import pytest

from src.tracking.tracker import ByteTrackAdapter, TrackedPerson


class MockTensor:
    """Small CPU/GPU-like wrapper with the conversion methods used by Torch."""

    def __init__(self, value):
        self.value = np.asarray(value)
        self.cpu_called = False

    def cpu(self):
        self.cpu_called = True
        return self

    def numpy(self):
        return self.value


class MockModel:
    def __init__(self, results):
        self.results = results
        self.call_args = None

    def track(self, frame, **kwargs):
        self.call_args = (frame, kwargs)
        return self.results


def _result(keypoints, confidences=None, ids=None):
    pose = SimpleNamespace(xy=keypoints, conf=confidences)
    boxes = None if ids is None else SimpleNamespace(id=ids)
    return SimpleNamespace(keypoints=pose, boxes=boxes)


def test_adapter_extracts_ids_keypoints_and_confidence_aligned_by_index():
    points = MockTensor(np.arange(2 * 17 * 2).reshape(2, 17, 2))
    confidence = MockTensor(np.full((2, 17), 0.85))
    ids = MockTensor(np.array([4, 9]))
    model = MockModel([_result(points, confidence, ids)])
    frame = object()

    people = ByteTrackAdapter(model, confidence=0.6).track(frame)

    assert len(people) == 2
    assert all(isinstance(person, TrackedPerson) for person in people)
    assert [person.track_id for person in people] == [4, 9]
    np.testing.assert_array_equal(people[0].keypoints, points.value[0])
    np.testing.assert_array_equal(people[1].confidence, confidence.value[1])
    assert points.cpu_called and confidence.cpu_called and ids.cpu_called
    assert model.call_args[0] is frame
    assert model.call_args[1] == {
        "persist": True,
        "tracker": "bytetrack.yaml",
        "conf": 0.6,
        "verbose": False,
    }


def test_adapter_preserves_missing_tracker_ids_as_none():
    model = MockModel([_result(np.zeros((2, 17, 2)), np.ones((2, 17)))])

    people = ByteTrackAdapter(model, confidence=0.5).track(None)

    assert [person.track_id for person in people] == [None, None]


@pytest.mark.parametrize(
    "results",
    [
        [],
        [SimpleNamespace(keypoints=None, boxes=None)],
        [_result(np.empty((0, 17, 2)), np.empty((0, 17)), np.empty(0))],
    ],
)
def test_adapter_returns_empty_list_for_empty_results(results):
    assert ByteTrackAdapter(MockModel(results), confidence=0.5).track(None) == []


def test_adapter_uses_none_ids_when_id_and_pose_counts_do_not_match():
    model = MockModel(
        [_result(np.zeros((2, 17, 2)), np.ones((2, 17)), np.array([77]))]
    )

    people = ByteTrackAdapter(model, confidence=0.5).track(None)

    assert [person.track_id for person in people] == [None, None]


def test_adapter_uses_no_confidence_when_confidence_count_does_not_match():
    model = MockModel(
        [_result(np.zeros((2, 17, 2)), np.ones((1, 17)), np.array([7, 8]))]
    )

    people = ByteTrackAdapter(model, confidence=0.5).track(None)

    assert [person.confidence for person in people] == [None, None]


def test_adapter_handles_id_values_that_cannot_be_converted_to_int():
    ids = np.array([3.0, np.nan])
    model = MockModel([_result(np.zeros((2, 17, 2)), ids=ids)])

    people = ByteTrackAdapter(model, confidence=0.5).track(None)

    assert [person.track_id for person in people] == [3, None]
