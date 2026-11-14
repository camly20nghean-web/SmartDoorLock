from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest

from scripts.demo_face_palm_unlock import FacePalmDemo, run_demo, seed_demo
from src.database.connection import DatabaseConnection
from src.database.models import AccessDecision, ActuationStatus


@pytest.fixture
def demo(tmp_path):
    service = FacePalmDemo(DatabaseConnection(str(tmp_path / "demo.sqlite3")))
    yield service
    service.lock.lock()


def test_demo_denials_never_call_unlock_and_grant_is_committed_first(demo, monkeypatch):
    unlock = demo.lock.unlock
    calls = []

    def checked_unlock():
        with closing(demo.db.get_connection()) as conn:
            row = conn.execute(
                "SELECT decision, actuation_status FROM access_logs ORDER BY log_id DESC LIMIT 1"
            ).fetchone()
        assert tuple(row) == ("GRANTED", "NOT_ACTUATED")
        calls.append(True)
        return unlock()

    monkeypatch.setattr(demo.lock, "unlock", checked_unlock)
    run_demo(demo)

    assert len(calls) == 1
    with closing(demo.db.get_connection()) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM access_logs WHERE actuation_status = 'ACTUATED'"
        ).fetchone()[0] == 1


def test_right_hand_and_concurrent_replay(demo):
    palm = seed_demo(demo)["DEMO_A"]
    demo.enroll_palm("DEMO_A", "RIGHT", palm)
    face = demo.face_templates["DEMO_A"]
    session_id = demo.issue_session()

    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = [pool.submit(demo.authenticate, session_id, face, palm, "RIGHT") for _ in range(2)]
        outcomes = [attempt.result() for attempt in attempts]

    assert {outcome.decision for outcome in outcomes} == {AccessDecision.GRANTED, AccessDecision.DENIED}
    assert sum(outcome.actuation_status == ActuationStatus.ACTUATED for outcome in outcomes) == 1
