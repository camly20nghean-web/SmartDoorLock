"""ESP32 Relay actuator controller via GPIO/UART/MQTT.

NOTE: This module is reserved for Phase 6 when physical 5V Relay and 12V Solenoid lock
are connected to the ESP32.
"""
from typing import Optional
from .base import BaseDoorLock, LockState


class ESP32RelayController(BaseDoorLock):
    """
    Hardware door lock controller sending electrical unlock pulses via ESP32 GPIO / Relay.
    
    Scheduled for implementation in Phase 6.
    """

    def __init__(self, endpoint_url: str = "http://192.168.1.100/lock/control") -> None:
        self.endpoint_url = endpoint_url
        self._state = LockState.LOCKED

    def unlock(self, duration_seconds: float = 5.0) -> bool:
        """Send unlock signal to ESP32 relay."""
        raise NotImplementedError(
            "ESP32 Relay hardware controller is scheduled for Phase 6. "
            "Please use MockDoorLock for Phase 1."
        )

    def lock(self) -> bool:
        """Send lock signal to ESP32 relay."""
        raise NotImplementedError("ESP32 Relay controller scheduled for Phase 6.")

    def get_state(self) -> LockState:
        return self._state

