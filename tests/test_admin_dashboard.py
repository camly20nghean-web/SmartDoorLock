from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "ui" / "dashboard.py"
API = "http://127.0.0.1:8000"


def test_user_search_and_registration_metadata(monkeypatch):
    def get(url, **kwargs):
        assert url == "http://127.0.0.1:8123/users"
        params = kwargs["params"]
        rows = [{
            "user_id": "P01", "full_name": "An", "status": "ACTIVE",
            "has_face": True, "has_left_palm": True, "has_right_palm": False,
            "vector": "not a dashboard column",
        }] if params.get("status") != "INACTIVE" else []
        return httpx.Response(200, json=rows, request=httpx.Request("GET", url))

    monkeypatch.setenv("SMARTDOORLOCK_API_URL", "http://127.0.0.1:8123/")
    monkeypatch.setattr(httpx, "get", get)
    app = AppTest.from_file(str(DASHBOARD), default_timeout=15).run()
    row = app.dataframe[0].value.iloc[0].to_dict()
    assert row == {
        "Mã người dùng": "P01", "Họ tên": "An", "Trạng thái": "ACTIVE",
        "Khuôn mặt": "Đã đăng ký", "Tay trái": "Đã đăng ký", "Tay phải": "Chưa đăng ký",
    }

    original_get = get

    def assert_search(url, **kwargs):
        assert kwargs["params"] == {"q": "P01", "status": "INACTIVE"}
        return original_get(url, **kwargs)

    monkeypatch.setattr(httpx, "get", assert_search)
    app.text_input[0].set_value(" P01 ")
    app.selectbox(key="user_status").select("INACTIVE").run()
    assert not app.exception
    assert "Không có người dùng" in app.info[0].value


def test_log_pagination_and_filter_reset(monkeypatch):
    requests = []
    logs = [
        {"request_id": str(i), "user_id": "P01", "auth_method": "PALM" if i > 1 else "DUAL",
         "decision": "GRANTED" if i > 2 else "DENIED", "actuation_status": "NOT_ACTUATED",
         "reason": "Demo", "timestamp_local": f"12/11/2026 14:00:0{i}"}
        for i in range(7, 0, -1)
    ]

    def get(url, **kwargs):
        params = kwargs["params"]
        requests.append(params.copy())
        data = []
        if url == f"{API}/access-logs":
            rows = [
                row for row in logs
                if all(row[key] == params[key] for key in ("auth_method", "decision") if key in params)
            ]
            offset, limit = params["offset"], params["limit"]
            data = {"items": rows[offset:offset + limit], "offset": offset, "limit": limit, "total": len(rows)}
        return httpx.Response(200, json=data, request=httpx.Request("GET", url))

    monkeypatch.delenv("SMARTDOORLOCK_API_URL", raising=False)
    monkeypatch.setattr(httpx, "get", get)
    app = AppTest.from_file(str(DASHBOARD), default_timeout=15).run()
    app.radio(key="page").set_value("Lịch sử xác thực").run()
    assert app.button(key="log_previous").disabled
    row = app.dataframe[0].value.iloc[0]
    assert row["Thời gian"] == "12/11/2026 14:00:07"
    assert row["Quyết định"] == "GRANTED" and row["Kết quả chấp hành"] == "NOT_ACTUATED"

    app.button(key="log_next").click().run()
    assert requests[-1] == {"limit": 5, "offset": 5}
    assert app.button(key="log_next").disabled
    app.button(key="log_previous").click().run()
    assert requests[-1]["offset"] == 0
    app.button(key="log_next").click().run()

    app.selectbox(key="log_method").select("PALM").run()
    assert requests[-1] == {"limit": 5, "offset": 0, "auth_method": "PALM"}
    app.selectbox(key="log_decision").select("DENIED").run()
    assert requests[-1] == {"limit": 5, "offset": 0, "auth_method": "PALM", "decision": "DENIED"}
    assert app.dataframe[0].value["Mã yêu cầu"].tolist() == ["2"]
    assert not app.exception


@pytest.mark.parametrize("unavailable", [True, False])
def test_api_error_shows_message(monkeypatch, unavailable):
    def get(url, **kwargs):
        request = httpx.Request("GET", url)
        if unavailable:
            raise httpx.ConnectError("Connection refused", request=request)
        return httpx.Response(503, request=request)

    monkeypatch.setattr(httpx, "get", get)
    app = AppTest.from_file(str(DASHBOARD), default_timeout=15).run()
    assert not app.exception
    expected = "Không kết nối được FastAPI" if unavailable else "API trả lỗi HTTP 503"
    assert expected in app.error[0].value
