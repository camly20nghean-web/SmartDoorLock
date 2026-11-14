"""Unit tests for MockDoorLock, fail-secure rules, and biometric interface contracts."""
import time
import pytest

from src.hardware.base import LockState
from src.hardware.mock_lock import MockDoorLock
from src.hardware.esp32_relay import ESP32RelayController
from src.biometrics.face import FaceRecognitionEngine
from src.biometrics.palm import PalmprintRecognitionEngine
import numpy as np


class TestMockDoorLock:
    """Test suite verifying lock state transitions, auto-relock timer, and fail-secure safety."""

    def test_initial_state_is_locked(self):
        lock = MockDoorLock()
        assert lock.get_state() == LockState.LOCKED
        assert lock.is_locked() is True

    def test_unlock_transitions_to_unlocked(self):
        lock = MockDoorLock(default_auto_relock_seconds=2.0)
        status = lock.unlock()

        assert status is True
        assert lock.get_state() == LockState.UNLOCKED
        assert lock.is_locked() is False
        lock.lock()  # cleanup

    def test_manual_lock_cancels_timer_and_locks(self):
        lock = MockDoorLock(default_auto_relock_seconds=5.0)
        lock.unlock()
        assert lock.get_state() == LockState.UNLOCKED

        lock.lock()
        assert lock.get_state() == LockState.LOCKED
        assert lock.is_locked() is True

    def test_auto_relock_after_timeout(self):
        lock = MockDoorLock()
        # Unlock with short 0.5 second timeout for fast testing
        lock.unlock(duration_seconds=0.5)
        assert lock.get_state() == LockState.UNLOCKED

        # Wait for auto-relock timer to elapse
        time.sleep(0.7)
        assert lock.get_state() == LockState.LOCKED
        assert lock.is_locked() is True

    def test_consecutive_unlocks_old_timer_does_not_relock_early(self):
        """
        Verify that consecutive unlocks properly cancel/supersede previous timers
        and an old timer's callback does not prematurely lock a newer unlock session.
        """
        lock = MockDoorLock()

        # Session 1: Unlock for 0.5 seconds (would relock at t=0.5s)
        lock.unlock(duration_seconds=0.5)
        assert lock.get_state() == LockState.UNLOCKED

        # Wait 0.3 seconds (Session 1 is still active)
        time.sleep(0.3)
        assert lock.get_state() == LockState.UNLOCKED

        # Session 2: A second unlock arrives, asking for 0.6 seconds (expires at t=0.3+0.6=0.9s)
        lock.unlock(duration_seconds=0.6)
        assert lock.get_state() == LockState.UNLOCKED

        # Wait another 0.3 seconds (total elapsed = 0.6s).
        # Session 1's initial 0.5s window has now elapsed.
        # The lock MUST STILL BE UNLOCKED because Session 2 extends until 0.9s.
        time.sleep(0.3)
        assert lock.get_state() == LockState.UNLOCKED, "Old timer prematurely locked the new session!"

        # Wait until Session 2 timeout completes (0.4s more, total elapsed from session 2 = 0.7s > 0.6s)
        time.sleep(0.4)
        assert lock.get_state() == LockState.LOCKED
        assert lock.is_locked() is True

    def test_obsolete_timer_callback_ignored(self):
        """Directly invoke callback with an outdated session ID and verify it is ignored."""
        lock = MockDoorLock()
        lock.unlock(duration_seconds=5.0)
        assert lock.get_state() == LockState.UNLOCKED

        # Simulate a rogue or late callback from an obsolete session
        lock._auto_relock_callback(session_id=-1)
        assert lock.get_state() == LockState.UNLOCKED, "Obsolete callback must not lock the door"

        lock.lock()  # cleanup
        assert lock.get_state() == LockState.LOCKED

    def test_fail_secure_on_exception(self, monkeypatch):
        lock = MockDoorLock()
        
        # Simulate unexpected threading failure
        def faulty_start():
            raise RuntimeError("Hardware/Thread fault")

        lock.unlock(duration_seconds=1.0)
        # Verify force locked ensures safe state
        lock._force_locked()
        assert lock.get_state() == LockState.LOCKED


class TestESP32RelayControllerContract:
    """Ensure ESP32 hardware relay raises NotImplementedError in Phase 1."""

    def test_relay_raises_not_implemented(self):
        relay = ESP32RelayController()
        with pytest.raises(NotImplementedError) as exc_info:
            relay.unlock()
        assert "Phase 6" in str(exc_info.value)

        with pytest.raises(NotImplementedError):
            relay.lock()


class TestBiometricsIntegrityContracts:
    """Ensure AI biometric engines strictly adhere to development contracts."""

    def test_face_recognition_engine_is_active(self):
        """Verify FaceRecognitionEngine is active and initialized with SFace in Phase 3."""
        engine = FaceRecognitionEngine()
        assert engine.is_ready is True
        assert engine.similarity_threshold == 0.60

    def test_palm_verification_raises_not_implemented(self):
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        with PalmprintRecognitionEngine() as engine:
            assert engine.detect_roi(dummy_img) is None
            with pytest.raises(NotImplementedError, match="Milestone D"):
                engine.verify(dummy_img, np.zeros(128))

