"""Exercise FACE_AND_PALM with synthetic inputs and a fresh demo SQLite file."""
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
from threading import Lock
import time
from uuid import uuid4

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.biometrics.palm import (
    DEFAULT_ROI_SIZE,
    PALM_FEATURE_DIMENSION,
    PALM_FEATURE_VERSION,
    PalmprintRecognitionEngine,
)
from src.database.connection import DatabaseConnection
from src.database.models import (
    AccessDecision,
    ActuationStatus,
    AuthMethod,
    AuthorizationOutcome,
    AuthPolicy,
    BiometricAuthClaim,
    BiometricType,
)
from src.database.repository import AccessLogRepository, UserRepository
from src.hardware.mock_lock import MockDoorLock
from src.services.access_control import AccessControlService

DEMO_PALM_THRESHOLD = 0.90
DEMO_FACE_THRESHOLD = 0.90


def unit_vector(vector, dimension):
    vector = np.asarray(vector, dtype="<f4")
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError("Invalid biometric vector.")
    norm = np.linalg.norm(vector.astype(np.float64))
    if norm == 0:
        raise ValueError("Zero biometric vector.")
    return (vector / norm).astype("<f4")


class FacePalmDemo:
    def __init__(self, db, *, face_threshold=DEMO_FACE_THRESHOLD):
        self.db = db
        db.init_db()
        self.users = UserRepository(db)
        self.logs = AccessLogRepository(db)
        self.policy = AccessControlService(self.users, self.logs)
        self.lock = MockDoorLock()
        self.sessions = {}
        self.session_lock = Lock()
        self.face_templates = {}
        self.face_threshold = face_threshold
        with closing(db.get_connection()) as conn, conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS demo_palm_templates (
                    user_id TEXT NOT NULL REFERENCES users(user_id),
                    hand_side TEXT NOT NULL CHECK (hand_side IN ('LEFT', 'RIGHT')),
                    feature_version TEXT NOT NULL,
                    vector BLOB NOT NULL CHECK (length(vector) = 16384),
                    PRIMARY KEY (user_id, hand_side, feature_version)
                )
            """)

    def enroll_palm(self, user_id, hand_side, vector, version=PALM_FEATURE_VERSION):
        vector = unit_vector(vector, PALM_FEATURE_DIMENSION)
        with closing(self.db.get_connection()) as conn, conn:
            conn.execute(
                "INSERT INTO demo_palm_templates VALUES (?, ?, ?, ?)",
                (user_id, hand_side, version, vector.tobytes()),
            )

    def palm_templates(self, hand_side):
        with closing(self.db.get_connection()) as conn:
            rows = conn.execute(
                "SELECT user_id, vector FROM demo_palm_templates "
                "WHERE hand_side = ? AND feature_version = ?",
                (hand_side, PALM_FEATURE_VERSION),
            ).fetchall()
        return {
            row["user_id"]: unit_vector(np.frombuffer(row["vector"], dtype="<f4"), PALM_FEATURE_DIMENSION)
            for row in rows
        }

    def issue_session(self, ttl_seconds=30):
        session_id = uuid4().hex
        with self.session_lock:
            self.sessions[session_id] = time.monotonic() + ttl_seconds
        return session_id

    @staticmethod
    def claim(session_id, modality, probe, templates, dimension, threshold):
        if probe is None:
            return None
        vector = unit_vector(probe, dimension)
        scores = sorted(
            ((float(np.clip(vector @ template, -1, 1)), user_id) for user_id, template in templates.items()),
            reverse=True,
        )
        score, user_id = scores[0] if scores else (0.0, None)
        matched = score >= threshold and (len(scores) == 1 or score > scores[1][0])
        return BiometricAuthClaim(
            user_id=user_id if matched else None,
            biometric_type=modality,
            is_matched=matched,
            similarity_score=score,
            session_id=session_id,
            timestamp=time.time(),
            is_trusted_source=True,
        )

    def authenticate(
        self, session_id, face_probe=None, palm_probe=None, hand_side=None,
        *, policy=AuthPolicy.FACE_AND_PALM,
    ):
        auth_method = {
            AuthPolicy.FACE_ONLY: AuthMethod.FACE,
            AuthPolicy.FACE_AND_PALM: AuthMethod.DUAL,
            AuthPolicy.PALM_ONLY: AuthMethod.PALM,
        }[policy]

        def outcome(decision, status, reason, user_id=None):
            return AuthorizationOutcome(session_id, decision, status, user_id, policy, reason)

        def deny(reason):
            return outcome(AccessDecision.DENIED, ActuationStatus.NOT_ACTUATED, reason)

        with self.session_lock:
            deadline = self.sessions.pop(session_id, None)
        if deadline is None:
            return deny("Unknown or already used session.")
        if time.monotonic() >= deadline:
            decision, user_id, reason = AccessDecision.DENIED, None, "Session expired."
        else:
            try:
                face = None
                if policy != AuthPolicy.PALM_ONLY:
                    face = self.claim(
                        session_id, BiometricType.FACE, face_probe, self.face_templates, 128, self.face_threshold,
                    )
                palm = self.claim(
                    session_id, BiometricType.PALM, palm_probe,
                    self.palm_templates(hand_side) if palm_probe is not None else {},
                    PALM_FEATURE_DIMENSION, DEMO_PALM_THRESHOLD,
                )
                decision, user_id, reason = self.policy.evaluate_authorization(policy, face, palm)
            except (sqlite3.Error, ValueError) as error:
                return deny(f"Unable to evaluate biometric inputs: {error}")

            if time.monotonic() >= deadline:
                decision, user_id, reason = AccessDecision.DENIED, None, "Session expired."
        try:
            log_id = self.logs.log_event(
                session_id, auth_method, decision, ActuationStatus.NOT_ACTUATED, reason, user_id,
                hand_side=hand_side if hand_side in ("LEFT", "RIGHT") else None,
            )
        except sqlite3.Error as error:
            return deny(f"Decision audit failed: {error}")

        if decision == AccessDecision.DENIED:
            return outcome(decision, ActuationStatus.NOT_ACTUATED, reason, user_id)

        # The committed audit row reserves the session before the mock can unlock.
        status = ActuationStatus.NOT_ACTUATED
        if time.monotonic() < deadline:
            status = ActuationStatus.ACTUATED if self.lock.unlock() else ActuationStatus.FAILED
        else:
            reason = "Session expired before actuation."
        try:
            with closing(self.db.get_connection()) as conn, conn:
                conn.execute(
                    "UPDATE access_logs SET actuation_status = ?, reason = ? WHERE log_id = ?",
                    (status.value, reason, log_id),
                )
        except sqlite3.Error as error:
            self.lock.lock()
            return outcome(decision, ActuationStatus.FAILED, f"Actuation audit failed: {error}", user_id)
        return outcome(decision, status, reason, user_id)


def seed_demo(demo):
    """Enroll controlled inputs through the real LBP extractor."""
    stripes = np.tile((np.arange(DEFAULT_ROI_SIZE) // 4 % 2 * 160 + 40).astype(np.uint8),
                      (DEFAULT_ROI_SIZE, 1))
    palm = {}
    with PalmprintRecognitionEngine() as engine:
        for index, (user_id, texture) in enumerate((("DEMO_A", stripes), ("DEMO_B", stripes.T))):
            demo.users.create_user(user_id, user_id)
            demo.face_templates[user_id] = np.eye(2, 128, dtype=np.float32)[index]
            roi = np.repeat(texture[:, :, None], 3, axis=2)
            palm[user_id] = engine.extract_embedding(roi)
            demo.enroll_palm(user_id, "LEFT", palm[user_id])
    return palm


def run_demo(demo):
    palm = seed_demo(demo)
    face = demo.face_templates["DEMO_A"]
    left = palm["DEMO_A"]

    def attempt(label, session_id, face_probe=face, palm_probe=left, hand="LEFT", granted=False):
        demo.lock.lock()
        result = demo.authenticate(session_id, face_probe, palm_probe, hand)
        expected = AccessDecision.GRANTED if granted else AccessDecision.DENIED
        expected_status = ActuationStatus.ACTUATED if granted else ActuationStatus.NOT_ACTUATED
        if result.decision != expected or result.actuation_status != expected_status or demo.lock.is_locked() == granted:
            raise RuntimeError(f"{label}: unexpected outcome {result}")
        print(f"{label}: {result.decision.value} / {result.actuation_status.value} - {result.reason}")

    session_id = demo.issue_session()
    attempt("Match", session_id, granted=True)
    attempt("Replay", session_id)
    attempt("Missing face", demo.issue_session(), face_probe=None)
    attempt("Missing palm", demo.issue_session(), palm_probe=None)
    attempt("Different people", demo.issue_session(), palm_probe=palm["DEMO_B"])
    attempt("Wrong hand", demo.issue_session(), hand="RIGHT")
    demo.enroll_palm("DEMO_A", "RIGHT", left, version="obsolete-demo-version")
    attempt("Wrong feature version", demo.issue_session(), hand="RIGHT")
    attempt("Expired session", demo.issue_session(ttl_seconds=0))

    demo.users.suspend_user("DEMO_A")
    attempt("Suspended account", demo.issue_session())
    demo.users.activate_user("DEMO_A")

    reserved = demo.issue_session()
    demo.logs.log_event(
        reserved, AuthMethod.DUAL, AccessDecision.DENIED, ActuationStatus.NOT_ACTUATED, "Demo reservation",
    )
    attempt("Duplicate audit ID", reserved)
    with closing(demo.db.get_connection()) as conn, conn:
        conn.execute("""
            CREATE TRIGGER demo_fail_audit BEFORE INSERT ON access_logs
            BEGIN SELECT RAISE(ABORT, 'simulated audit failure'); END
        """)
        try:
            attempt("Audit write failure", demo.issue_session())
        finally:
            conn.execute("DROP TRIGGER demo_fail_audit")


def main():
    db_path = Path(tempfile.mkdtemp(prefix="smartdoorlock-milestone-d-")) / "demo.sqlite3"
    print(f"Demo database: {db_path}")
    print(f"Synthetic inputs; palm cosine threshold: {DEMO_PALM_THRESHOLD:.2f}")
    demo = FacePalmDemo(DatabaseConnection(str(db_path)))
    try:
        run_demo(demo)
    finally:
        demo.lock.lock()


if __name__ == "__main__":
    main()
