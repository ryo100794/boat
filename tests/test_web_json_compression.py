import gzip
from io import BytesIO
import json

from boatrace_ai.web.dashboard import (
    _read_json_artifact,
    _read_json_artifact_cached,
    model_performance_daily_report,
    model_performance_public_report,
    send_json,
)


class ResponseRecorder:
    def __init__(self, accept_encoding: str = "") -> None:
        self.headers = {"Accept-Encoding": accept_encoding}
        self.response_status = None
        self.response_headers: dict[str, str] = {}
        self.wfile = BytesIO()

    def send_response(self, status: int) -> None:
        self.response_status = status

    def send_header(self, name: str, value: str) -> None:
        self.response_headers[name] = value

    def end_headers(self) -> None:
        return None


def test_json_artifact_cache_reloads_changed_file(tmp_path) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"value":1}', encoding="utf-8")
    _read_json_artifact_cached.cache_clear()

    first = _read_json_artifact(path)
    cached = _read_json_artifact(path)
    assert cached is first

    path.write_text('{"value":200}', encoding="utf-8")
    updated = _read_json_artifact(path)

    assert updated == {"value": 200}
    assert updated is not first


def test_send_json_compresses_large_payload_when_client_accepts_gzip() -> None:
    handler = ResponseRecorder("br, gzip")
    payload = {"rows": [{"value": "x" * 200}] * 100}

    send_json(handler, payload)

    body = handler.wfile.getvalue()
    assert handler.response_status == 200
    assert handler.response_headers["Content-Encoding"] == "gzip"
    assert handler.response_headers["Vary"] == "Accept-Encoding"
    assert int(handler.response_headers["Content-Length"]) == len(body)
    assert json.loads(gzip.decompress(body)) == payload


def test_send_json_leaves_small_payload_uncompressed() -> None:
    handler = ResponseRecorder("gzip")

    send_json(handler, {"ok": True})

    assert "Content-Encoding" not in handler.response_headers
    assert json.loads(handler.wfile.getvalue()) == {"ok": True}


def test_public_model_report_omits_duplicated_legacy_daily_rows() -> None:
    report = {
        "bankroll_daily": {"legacy": [{"race_date": "2026-07-24"}]},
        "model_daily": {
            "current": {"rows": [{"date": "2026-07-24"}]},
            "missing": {"rows": [], "unavailable_reason": "日次なし"},
        },
        "bankroll": [],
    }

    public = model_performance_public_report(report)

    assert "bankroll_daily" not in public
    assert public["model_daily"]["current"] == {
        "row_count": 1,
        "loaded": False,
    }
    assert public["model_daily"]["missing"] == {
        "row_count": 0,
        "loaded": True,
        "rows": [],
        "unavailable_reason": "日次なし",
    }
    assert "bankroll_daily" in report
    assert report["model_daily"]["current"]["rows"]


def test_model_daily_report_returns_only_requested_model() -> None:
    report = {
        "model_daily": {
            "alpha": {"rows": [{"date": "2026-07-24"}]},
            "beta": {"rows": [{"date": "2026-07-25"}]},
        }
    }

    daily = model_performance_daily_report(report, {"model_key": ["alpha"]})

    assert daily == {
        "loaded": True,
        "rows": [{"date": "2026-07-24"}],
    }
    missing = model_performance_daily_report(report, {"model_key": ["unknown"]})
    assert missing["loaded"] is True
    assert missing["rows"] == []
    assert "指定モデル" in missing["unavailable_reason"]
