"""Mock door lock actuator for testing and development environments."""
import logging
import threading
from typing import Optional

from .base import BaseDoorLock, LockState

logger = logging.getLogger(__name__)


class MockDoorLock(BaseDoorLock):
    """
    Simulated electronic door lock for development and testing.
    
    Provides simulated state transitions (LOCKED / UNLOCKED) with thread-safe
    auto-relocking.
    
    IMPORTANT LIMITATION:
    This class is a software mock running in user-space memory. It enforces
    logical fail-secure behavior in code (defaulting to LOCKED upon initialization
    or software exceptions). It CANNOT and DOES NOT guarantee physical locking
    during physical power outages or OS crashes; physical fail-secure security
    is strictly a hardware property of the physical 12V solenoid/relay (Phase 6).
    """

    def __init__(self, default_auto_relock_seconds: float = 5.0) -> None:
        self.default_auto_relock_seconds = max(1.0, default_auto_relock_seconds)
        self._state = LockState.LOCKED
        self._mutex = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._session_id: int = 0
        logger.info("[MockDoorLock] Initialized in safe state: %s", self._state.value)

    def unlock(self, duration_seconds: Optional[float] = None) -> bool:
        """
        Transition lock to UNLOCKED for duration_seconds, then auto-relock.
        
        Args:
            duration_seconds (Optional[float]): Custom timeout or None to use default.
            
        Returns:
            bool: True if unlocked successfully, False if an error occurred.
        """
        timeout = duration_seconds if duration_seconds is not None else self.default_auto_relock_seconds
        timeout = max(0.5, float(timeout))

        try:
            with self._mutex:
                # Invalidate any previously running timer session
                self._session_id += 1
                current_session = self._session_id

                if self._timer is not None and self._timer.is_alive():
                    self._timer.cancel()
                    self._timer = None

                self._state = LockState.UNLOCKED
                logger.info(
                    "🔓 [MockDoorLock] DOOR UNLOCKED (Session #%d). Auto-relock scheduled in %.1f seconds.",
                    current_session, timeout
                )

                # Schedule background auto-relock tagged with current session id
                self._timer = threading.Timer(
                    timeout, self._auto_relock_callback, args=(current_session,)
                )
                self._timer.daemon = True
                self._timer.start()
                return True
        except Exception as e:
            logger.error("[MockDoorLock] Error during unlock; enforcing fail-secure LOCKED: %s", e)
            self._force_locked()
            return False

    def lock(self) -> bool:
        """
        Immediately lock the door and cancel any pending auto-relock timer.
        
        Returns:
            bool: True if locked successfully.
        """
        try:
            with self._mutex:
                self._session_id += 1
                if self._timer is not None and self._timer.is_alive():
                    self._timer.cancel()
                    self._timer = None

                self._state = LockState.LOCKED
                logger.info("🔒 [MockDoorLock] DOOR LOCKED.")
                return True
        except Exception as e:
            logger.error("[MockDoorLock] Exception while locking; forcing LOCKED: %s", e)
            self._force_locked()
            return True

    def get_state(self) -> LockState:
        """Return the current lock state."""
        with self._mutex:
            return self._state

    def _auto_relock_callback(self, session_id: int) -> None:
        """
        Internal callback invoked by timer to lock the door.
        
        Verifies session_id matches active session to prevent race conditions
        where an expired timer from a previous unlock prematurely locks a newer
        unlock session.
        """
        with self._mutex:
            if session_id != self._session_id:
                logger.debug(
                    "[MockDoorLock] Ignored obsolete timer for session #%d (active session is #%d).",
                    session_id, self._session_id
                )
                return

            logger.info("⏱️ [MockDoorLock] Auto-relock timeout elapsed for session #%d.", session_id)
            self._state = LockState.LOCKED
            self._timer = None

    def _force_locked(self) -> None:
        """Emergency fail-secure fallback enforcing LOCKED state."""
        self._session_id += 1
        self._state = LockState.LOCKED
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

