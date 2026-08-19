"""Tests for checkpoint/sync_checkpoint recovery from a pipeline restart.

Covers feldera/cloud#1927: `checkpoint(wait=True)`/`sync_checkpoint(wait=True)`
must not hang forever if the pipeline process restarts between the request
and a status poll. These mock the `FelderaClient` so they exercise the
`Pipeline` retry logic directly, without a running pipeline.
"""

from types import SimpleNamespace
from unittest import mock

import requests

from feldera.pipeline import Pipeline
from feldera.rest.errors import FelderaAPIError

INCARNATION_A = "aaaaaaaa-0000-0000-0000-000000000000"
INCARNATION_B = "bbbbbbbb-0000-0000-0000-000000000000"


def _pipeline(client) -> Pipeline:
    pipeline = Pipeline(client)
    pipeline._inner = SimpleNamespace(name="test_pipeline")
    return pipeline


def _incarnation_mismatch_error() -> FelderaAPIError:
    resp = requests.Response()
    resp.status_code = 400
    resp._content = b'{"error_code": "IncarnationUuidMismatch", "message": "restarted"}'
    resp.headers["content-type"] = "application/json"
    prepared = requests.PreparedRequest()
    prepared.prepare(method="GET", url="http://example.test/v0/x")
    resp.request = prepared
    return FelderaAPIError("mismatch", resp)


class TestCheckpointRestartRecovery:
    def test_retries_transparently_on_incarnation_mismatch(self):
        client = mock.Mock()
        client.checkpoint_pipeline.side_effect = [
            {"checkpoint_sequence_number": 1, "incarnation_uuid": INCARNATION_A},
            {"checkpoint_sequence_number": 2, "incarnation_uuid": INCARNATION_B},
        ]
        client.checkpoint_pipeline_status.side_effect = [
            _incarnation_mismatch_error(),
            {"success": 2, "failure": None},
        ]

        with mock.patch("feldera.pipeline.time.sleep"):
            seq = _pipeline(client).checkpoint(wait=True)

        assert seq == 2
        assert client.checkpoint_pipeline.call_count == 2
        client.checkpoint_pipeline_status.assert_has_calls(
            [
                mock.call("test_pipeline", INCARNATION_A),
                mock.call("test_pipeline", INCARNATION_B),
            ]
        )

    def test_sleeps_between_mismatch_retries(self):
        """A crash-looping pipeline must not turn this into a tight loop."""
        client = mock.Mock()
        client.checkpoint_pipeline.side_effect = [
            {"checkpoint_sequence_number": 1, "incarnation_uuid": INCARNATION_A},
            {"checkpoint_sequence_number": 2, "incarnation_uuid": INCARNATION_B},
        ]
        client.checkpoint_pipeline_status.side_effect = [
            _incarnation_mismatch_error(),
            {"success": 2, "failure": None},
        ]

        with mock.patch("feldera.pipeline.time.sleep") as sleeper:
            _pipeline(client).checkpoint(wait=True)

        assert sleeper.call_count >= 1

    def test_handles_missing_incarnation_uuid_from_older_pipeline(self):
        """A pipeline predating this field omits `incarnation_uuid`; the SDK
        must fall back to the old polling behavior instead of raising."""
        client = mock.Mock()
        client.checkpoint_pipeline.return_value = {"checkpoint_sequence_number": 5}
        client.checkpoint_pipeline_status.return_value = {
            "success": 5,
            "failure": None,
        }

        seq = _pipeline(client).checkpoint(wait=True)

        assert seq == 5
        client.checkpoint_pipeline_status.assert_called_once_with("test_pipeline", None)

    def test_non_mismatch_api_error_still_propagates(self):
        client = mock.Mock()
        client.checkpoint_pipeline.return_value = {
            "checkpoint_sequence_number": 1,
            "incarnation_uuid": INCARNATION_A,
        }

        resp = requests.Response()
        resp.status_code = 500
        resp._content = b'{"error_code": "SomeOtherError", "message": "boom"}'
        resp.headers["content-type"] = "application/json"
        prepared = requests.PreparedRequest()
        prepared.prepare(method="GET", url="http://example.test/v0/x")
        resp.request = prepared
        client.checkpoint_pipeline_status.side_effect = FelderaAPIError("boom", resp)

        try:
            _pipeline(client).checkpoint(wait=True)
            raise AssertionError("expected FelderaAPIError to propagate")
        except FelderaAPIError as e:
            assert e.error_code == "SomeOtherError"


class TestSyncCheckpointRestartRecovery:
    def test_retries_transparently_on_incarnation_mismatch(self):
        client = mock.Mock()
        client.sync_checkpoint.side_effect = [
            {
                "checkpoint_uuid": "11111111-0000-0000-0000-000000000000",
                "incarnation_uuid": INCARNATION_A,
            },
            {
                "checkpoint_uuid": "22222222-0000-0000-0000-000000000000",
                "incarnation_uuid": INCARNATION_B,
            },
        ]
        client.sync_checkpoint_status.side_effect = [
            _incarnation_mismatch_error(),
            {
                "success": "22222222-0000-0000-0000-000000000000",
                "periodic": None,
                "failure": None,
            },
        ]

        with mock.patch("feldera.pipeline.time.sleep"):
            uuid = _pipeline(client).sync_checkpoint(wait=True)

        assert uuid == "22222222-0000-0000-0000-000000000000"
        assert client.sync_checkpoint.call_count == 2
        client.sync_checkpoint_status.assert_has_calls(
            [
                mock.call("test_pipeline", INCARNATION_A),
                mock.call("test_pipeline", INCARNATION_B),
            ]
        )

    def test_handles_missing_incarnation_uuid_from_older_pipeline(self):
        client = mock.Mock()
        client.sync_checkpoint.return_value = {
            "checkpoint_uuid": "33333333-0000-0000-0000-000000000000"
        }
        client.sync_checkpoint_status.return_value = {
            "success": "33333333-0000-0000-0000-000000000000",
            "periodic": None,
            "failure": None,
        }

        uuid = _pipeline(client).sync_checkpoint(wait=True)

        assert uuid == "33333333-0000-0000-0000-000000000000"
        client.sync_checkpoint_status.assert_called_once_with("test_pipeline", None)
