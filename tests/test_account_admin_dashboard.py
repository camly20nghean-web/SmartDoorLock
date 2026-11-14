"""Exercise dashboard forms with the real API and a disposable SQLite file."""
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
from streamlit.testing.v1 import AppTest

from scripts.demo_face_palm_unlock import FacePalmDemo
from scripts.prepare_account_admin_demo import prepare
from src.api.main import create_app
from src.database.connection import DatabaseConnection

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "ui" / "dashboard.py"


def test_account_management_forms(tmp_path, monkeypatch):
    original = tmp_path / "original.sqlite3"
    demo = FacePalmDemo(DatabaseConnection(str(original)))
    demo.users.create_user("P01", "An")
    path = prepare(original, tmp_path / "shared.sqlite3")
    client = TestClient(create_app(path), client=("127.0.0.1", 50000))

    def api_request(method, url, **kwargs):
        kwargs.pop("trust_env")
        kwargs.pop("timeout")
        return client.request(method, httpx.URL(url).path, **kwargs)

    monkeypatch.setattr(httpx, "get", lambda url, **kwargs: api_request("GET", url, **kwargs))
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: api_request("POST", url, **kwargs))
    page = AppTest.from_file(str(DASHBOARD), default_timeout=15).run()

    def input_text(label, value):
        next(widget for widget in page.text_input if widget.label == label).set_value(value)

    def click(label):
        next(widget for widget in page.button if widget.label == label).click().run()
        assert not page.exception

    assert any("SQLite thử nghiệm dùng chung" in item.value for item in page.caption)
    input_text("Mã tài khoản mới", "P02")
    input_text("Họ tên", "Test")
    click("Tạo tài khoản")
    assert "P02" in page.dataframe[0].value["Mã người dùng"].tolist()

    next(w for w in page.selectbox if w.label == "Tài khoản cần đổi trạng thái").select("P01").run()
    click("Vô hiệu hóa")
    assert any("Mã xác nhận chưa khớp" in item.value for item in page.error)
    input_text("Nhập mã P01 để xác nhận vô hiệu hóa", "P01")
    click("Vô hiệu hóa")
    assert page.dataframe[0].value.iloc[0]["Trạng thái"] == "INACTIVE"
    click("Kích hoạt lại")
    assert page.dataframe[0].value.iloc[0]["Trạng thái"] == "ACTIVE"

