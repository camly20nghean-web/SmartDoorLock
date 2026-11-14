"""Streamlit dashboard backed by the SmartDoorLock admin API."""
import base64
import os

import httpx
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"
ALL = "Tất cả"
METHODS = (ALL, "FACE", "PALM", "DUAL", "MANUAL")
DECISIONS = (ALL, "GRANTED", "DENIED")
USER_STATUSES = (ALL, "ACTIVE", "INACTIVE", "SUSPENDED")


def fetch(api_url, endpoint, params):
    try:
        response = httpx.get(f"{api_url}{endpoint}", params=params, timeout=5.0, trust_env=False)
        response.raise_for_status()
        source = response.headers.get("X-SmartDoorLock-Source")
        if source:
            st.caption("Nguồn: SQLite thử nghiệm dùng chung · MockDoorLock" if source == "shared-demo"
                       else "Nguồn: SQLite chỉ đọc")
        st.session_state["account_management"] = (
            response.headers.get("X-SmartDoorLock-Account-Management") == "enabled"
        )
        return response.json()
    except httpx.HTTPStatusError as error:
        st.error(f"API trả lỗi HTTP {error.response.status_code}. Kiểm tra nhật ký FastAPI và cấu hình SQLite.")
    except (httpx.RequestError, httpx.InvalidURL):
        st.error(f"Không kết nối được FastAPI tại {api_url}. Kiểm tra địa chỉ và khởi động API rồi bấm Làm mới.")
    except ValueError:
        st.error("Phản hồi từ FastAPI không phải JSON hợp lệ.")
    st.stop()


def admin_request(api_url, endpoint, *, data=None, params=None):
    try:
        response = httpx.post(
            f"{api_url}{endpoint}", json=data, params=params,
            timeout=120.0 if endpoint in ("/admin/enrollment", "/admin/verify") else 12.0,
            trust_env=False,
        )
        response.raise_for_status()
        return response.json() if response.content else {}
    except httpx.HTTPStatusError as error:
        code = error.response.status_code
        if endpoint in ("/admin/enrollment", "/admin/verify", "/admin/camera-preview"):
            try:
                detail = error.response.json().get("detail")
            except ValueError:
                detail = None
            message = f"Thao tác chưa hoàn tất: {detail if isinstance(detail, str) else f'HTTP {code}'}"
        elif code == 409 and endpoint == "/admin/users":
            message = "Mã người dùng đã tồn tại."
        else:
            message = {
                403: "Nguồn này không cho phép thay đổi tài khoản.",
                404: "Không tìm thấy tài khoản. Hãy làm mới.",
                409: "Trạng thái tài khoản đã thay đổi hoặc không cho phép thao tác này.",
                422: "Thông tin không hợp lệ. Kiểm tra mã và họ tên.",
                503: "Không lưu được thay đổi; tài khoản và audit vẫn giữ nguyên.",
            }.get(code, f"API trả lỗi HTTP {code}.")
        st.error(message)
    except (httpx.RequestError, httpx.InvalidURL, ValueError):
        st.error("Không nhận được phản hồi hợp lệ. Làm mới để kiểm tra kết quả trước khi thử lại.")
    return None


def show_account_management(api_url, users):
    st.subheader("Quản lý tài khoản")
    if message := st.session_state.pop("account_notice", None):
        st.success(message)
    with st.form("create_account"):
        user_id = st.text_input("Mã tài khoản mới", max_chars=100)
        full_name = st.text_input("Họ tên", max_chars=200)
        if st.form_submit_button("Tạo tài khoản"):
            if admin_request(api_url, "/admin/users", data={"user_id": user_id, "full_name": full_name}):
                st.session_state["account_notice"] = f"Đã tạo tài khoản {user_id.strip()}."
                st.rerun()

    eligible = {user["user_id"]: user["status"] for user in users if user["status"] in ("ACTIVE", "INACTIVE")}
    if not eligible:
        return
    target = st.selectbox("Tài khoản cần đổi trạng thái", list(eligible), index=None)
    if target is None:
        return
    deactivate = eligible[target] == "ACTIVE"
    with st.form(f"change_account:{target}:{eligible[target]}", clear_on_submit=True):
        confirmation = st.text_input(f"Nhập mã {target} để xác nhận vô hiệu hóa", max_chars=100) if deactivate else None
        if st.form_submit_button("Vô hiệu hóa" if deactivate else "Kích hoạt lại"):
            if deactivate and confirmation != target:
                st.error("Mã xác nhận chưa khớp.")
                return
            result = admin_request(
                api_url, "/admin/user-status", params={"user_id": target},
                data={"action": "DEACTIVATE" if deactivate else "ACTIVATE", "confirmation": confirmation},
            )
            if result:
                st.session_state["account_notice"] = f"Tài khoản {target} đã chuyển sang {result['status']}."
                st.rerun()


def show_enrollment(api_url, users):
    st.subheader("Đăng ký sinh trắc học")
    if message := st.session_state.pop("enrollment_notice", None):
        st.success(message)
    active = {user["user_id"]: user for user in users if user["status"] == "ACTIVE"}
    if not active:
        st.info("Chọn bộ lọc có tài khoản đang hoạt động để đăng ký.")
        return
    user_id = st.selectbox("Tài khoản đăng ký sinh trắc học", list(active), index=None)
    if user_id is None:
        return
    options = {"FACE": ("Khuôn mặt", "has_face"), "LEFT": ("Tay trái", "has_left_palm"),
               "RIGHT": ("Tay phải", "has_right_palm")}
    sample = st.selectbox("Mẫu cần đăng ký", list(options), format_func=lambda key: options[key][0])
    label, field = options[sample]
    existing = active[user_id][field]
    replace = existing and st.checkbox(
        f"Xác nhận thay mẫu {label.lower()} của {user_id}", key=f"replace:{user_id}:{sample}",
    )
    st.caption("Trong cửa sổ webcam, nhấn Space để lấy từng mẫu đạt chất lượng. Thu đủ 5 mẫu hoặc nhấn Q/Esc để hủy.")
    if st.button(f"Đăng ký {label.lower()}", disabled=existing and not replace):
        with st.spinner("Đang thu mẫu. Chuyển sang cửa sổ webcam để xem tiến độ."):
            result = admin_request(
                api_url, "/admin/enrollment", params={"user_id": user_id},
                data={"sample": sample, "replace": replace},
            )
        if result:
            st.session_state["enrollment_notice"] = f"Đã lưu mẫu {label.lower()} cho {user_id}."
            st.rerun()


def show_users(api_url):
    st.header("Người dùng")
    search_column, status_column = st.columns([2, 1])
    query = search_column.text_input("Tìm theo mã hoặc tên", max_chars=100)
    status = status_column.selectbox("Trạng thái", USER_STATUSES, key="user_status")
    params = {}
    if query.strip():
        params["q"] = query.strip()
    if status != ALL:
        params["status"] = status

    users = fetch(api_url, "/users", params)
    if not users:
        st.info("Không có người dùng khớp bộ lọc.")
        if st.session_state["account_management"]:
            show_account_management(api_url, users)
        return

    st.caption(f"{len(users)} người dùng")
    st.dataframe([
        {
            "Mã người dùng": user["user_id"],
            "Họ tên": user["full_name"],
            "Trạng thái": user["status"],
            "Khuôn mặt": "Đã đăng ký" if user["has_face"] else "Chưa đăng ký",
            "Tay trái": "Đã đăng ký" if user["has_left_palm"] else "Chưa đăng ký",
            "Tay phải": "Đã đăng ký" if user["has_right_palm"] else "Chưa đăng ký",
        }
        for user in users
    ], hide_index=True, width="stretch")


    if st.session_state["account_management"]:
        show_account_management(api_url, users)
        show_enrollment(api_url, users)


def change_log_offset(offset):
    st.session_state["log_offset"] = offset


def show_logs(api_url):
    st.header("Lịch sử xác thực")
    method_column, decision_column, size_column = st.columns([1, 1, 1])
    method = method_column.selectbox(
        "Phương thức", METHODS, key="log_method", on_change=change_log_offset, args=(0,),
    )
    decision = decision_column.selectbox(
        "Quyết định", DECISIONS, key="log_decision", on_change=change_log_offset, args=(0,),
    )
    limit = size_column.selectbox(
        "Số bản ghi mỗi trang", (2, 5, 10, 20), index=1,
        key="log_limit", on_change=change_log_offset, args=(0,),
    )
    offset = st.session_state.setdefault("log_offset", 0)
    params = {"limit": limit, "offset": offset}
    if method != ALL:
        params["auth_method"] = method
    if decision != ALL:
        params["decision"] = decision

    page = fetch(api_url, "/access-logs", params)
    items, total = page["items"], page["total"]
    if total == 0:
        st.info("Không có bản ghi xác thực khớp bộ lọc.")
        return

    if items:
        st.caption(f"Bản ghi {offset + 1}–{offset + len(items)} / {total} · Giờ Việt Nam")
        st.dataframe([
            {
                "Thời gian": item.get("timestamp_local") or "—",
                "Mã yêu cầu": item["request_id"],
                "Người dùng": item["user_id"] or "—",
                "Phương thức": item["auth_method"],
                "Bên tay": {"LEFT": "Tay trái", "RIGHT": "Tay phải"}.get(
                    item.get("hand_side"),
                    "Chưa ghi nhận" if item["auth_method"] in ("PALM", "DUAL") else "Không áp dụng",
                ),
                "Quyết định": item["decision"],
                "Kết quả chấp hành": item["actuation_status"],
                "Lý do": item["reason"],
            }
            for item in items
        ], hide_index=True, width="stretch")
    else:
        st.info("Trang này không còn bản ghi. Chọn Trang trước để quay lại.")

    previous, following, _ = st.columns([1, 1, 3])
    previous.button(
        "Trang trước", key="log_previous", disabled=offset == 0,
        on_click=change_log_offset, args=(max(0, offset - limit),), width="stretch",
    )
    following.button(
        "Trang sau", key="log_next", disabled=offset + limit >= total,
        on_click=change_log_offset, args=(offset + limit,), width="stretch",
    )


def show_unlock_result(result, method, has_palm, names):
    decision = result["decision"]
    actuation = result["actuation_status"]
    account = result["user_id"]
    hand = {"LEFT": "Tay trái", "RIGHT": "Tay phải"}.get(
        result["hand_side"], "Chưa ghi nhận" if has_palm else "Không áp dụng",
    )
    name = names.get(account)

    with st.container(border=True):
        st.subheader("Kết quả lần thử")
        if decision == "GRANTED" and actuation == "ACTUATED":
            st.success("Khóa giả lập đã mở", icon=":material/lock_open:")
        elif decision == "GRANTED":
            st.warning("Đã cấp quyền nhưng khóa giả lập chưa mở", icon=":material/warning:")
        else:
            st.warning("Xác thực bị từ chối", icon=":material/lock:")

        permission, operation = st.columns(2)
        with permission:
            st.caption("Quyết định xác thực")
            st.text({"GRANTED": "Cấp quyền", "DENIED": "Từ chối"}.get(decision, decision))
        with operation:
            st.caption("Kết quả chấp hành")
            st.text({
                "ACTUATED": "Đã mở khóa giả lập",
                "NOT_ACTUATED": "Không thực hiện mở khóa",
                "FAILED": "Lệnh mở khóa thất bại",
            }.get(actuation, actuation))

        person, mode, side = st.columns([2, 2, 1])
        with person:
            st.caption("Người dùng")
            st.text(f"{name} ({account})" if name and name != account else account or "Chưa xác định")
        with mode:
            st.caption("Phương thức")
            st.text(method)
        with side:
            st.caption("Bên tay")
            st.text(hand)

        if result.get("capture_error"):
            st.caption("Thông báo từ webcam")
            st.text(result["capture_error"])
        st.caption("Lý do xác thực")
        st.text(result["reason"])
        with st.expander("Chi tiết để đối chiếu lịch sử"):
            st.caption("Mã yêu cầu")
            st.code(result["request_id"], language=None)
            st.caption(f"Mã kết quả · {decision} / {actuation}")


def show_unlock(api_url):
    st.header("Thử mở khóa")
    users = fetch(api_url, "/users", {})
    if not st.session_state["account_management"]:
        st.info("Chọn SQLite thử nghiệm dùng chung để thử mở khóa.")
        return
    source = st.radio("Nguồn camera", ("Webcam máy tính", "ESP32-CAM"), horizontal=True)
    selection = {"source": "webcam"}
    if source == "ESP32-CAM":
        url = st.text_input("Địa chỉ luồng hình", placeholder="http://192.168.1.50:81/stream", max_chars=512).strip()
        selection = {"source": "esp32", "url": url}
        if st.button("Kiểm tra hình ảnh", disabled=not url):
            with st.spinner("Đang lấy ảnh từ ESP32-CAM."):
                preview = admin_request(api_url, "/admin/camera-preview", data=selection)
            if preview:
                st.image(base64.b64decode(preview["jpeg"]), caption="Khung hình từ ESP32-CAM")
    modes = {
        "Khuôn mặt": "FACE_ONLY",
        "Lòng bàn tay": "PALM_ONLY",
        "Khuôn mặt + lòng bàn tay": "FACE_AND_PALM",
    }
    label = st.radio("Phương thức xác thực", list(modes))
    st.caption("Trong cửa sổ webcam, nhấn Space khi mẫu sẵn sàng. Nhấn Q hoặc Esc để hủy.")
    if st.button("Bắt đầu xác thực", type="primary", disabled=source == "ESP32-CAM" and not selection["url"]):
        st.session_state.pop("unlock_result", None)
        with st.spinner("Đang xác thực. Chuyển sang cửa sổ webcam để xem hướng dẫn."):
            result = admin_request(api_url, "/admin/verify", params={"policy": modes[label]}, data=selection)
        if result:
            st.session_state["unlock_result"] = (api_url, label, source, result)

    previous = st.session_state.get("unlock_result")
    if previous and len(previous) == 4 and previous[0] == api_url:
        _, method, used_source, result = previous
        st.caption(f"Nguồn camera của lần thử: {used_source}")
        names = {user["user_id"]: user["full_name"] for user in users}
        show_unlock_result(result, method, modes[method] != "FACE_ONLY", names)


def main():
    st.set_page_config(page_title="SmartDoorLock · Quản trị", layout="wide")
    api_url = (os.environ.get("SMARTDOORLOCK_API_URL") or DEFAULT_API_URL).strip().rstrip("/")
    with st.sidebar:
        st.header("SmartDoorLock")
        page = st.radio(
            "Màn hình", ("Người dùng", "Lịch sử xác thực", "Thử mở khóa"), key="page",
            on_change=change_log_offset, args=(0,),
        )
        st.divider()
        st.caption("Quản trị cục bộ")
        st.button("Làm mới")
    st.title("Quản trị SmartDoorLock")
    if page == "Người dùng":
        show_users(api_url)
    elif page == "Lịch sử xác thực":
        show_logs(api_url)
    else:
        show_unlock(api_url)


if __name__ == "__main__":
    main()
