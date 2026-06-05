# SPDX-License-Identifier: Apache-2.0

from sglang_omni.scheduling.prepared_request_store import (
    PreparedRequestStore,
    prepared_marker_from_data,
)


def test_prepared_request_store_pop_and_cleanup() -> None:
    store: PreparedRequestStore[object] = PreparedRequestStore()
    prepared = object()

    assert store.publish("req", prepared) is True
    assert store.pop("req") is prepared
    assert store.pop("req") is None

    assert store.publish("cleanup", prepared) is True
    store.cleanup("cleanup")
    assert store.pop("cleanup") is None


def test_prepared_request_store_drops_late_publish_after_inflight_abort() -> None:
    store: PreparedRequestStore[object] = PreparedRequestStore()
    prepared = object()

    store.cleanup("ghost")
    assert store.aborted == set()

    store.mark_inflight("req")
    store.cleanup("req")

    assert store.aborted == {"req"}
    assert store.publish("req", prepared) is False
    assert store.pop("req") is None
    assert store.inflight == set()
    assert store.aborted == set()


def test_prepared_request_store_error_cleanup_clears_tombstone() -> None:
    store: PreparedRequestStore[object] = PreparedRequestStore()

    store.mark_inflight("req")
    store.cleanup("req")
    store.discard_inflight_after_error("req")

    assert store.inflight == set()
    assert store.aborted == set()
    assert store.prepared == {}


def test_prepared_marker_from_data() -> None:
    assert prepared_marker_from_data({"marker": 7}, "marker") == "7"
    assert prepared_marker_from_data({"other": 7}, "marker") is None
    assert prepared_marker_from_data([], "marker") is None
