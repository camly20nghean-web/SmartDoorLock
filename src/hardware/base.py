"""Abstract Base Class and enumerations for door lock actuators."""
from abc import ABC, abstractmethod
from enum import Enum


class LockState(str, Enum):
    """Door lock actuator operational states."""
    LOCKED = "LOCKED"
    UNLOCKED = "UNLOCKED"


class BaseDoorLock(ABC):
    """
    Abstract Base Class for door locking actuators.
    
    Adheres to the Fail-Secure security principle:
    - In software: The default state is always LOCKED, and any unhandled
      software exception forces the lock state to LOCKED.
    - In physical hardware (Phase 6): The physical actuator must be wired such
      that total power loss keeps the door physically locked (e.g. normally-locked
      12V solenoid lock with spring return). Software simulations cannot enforce
      physical behavior during total power loss.
    """

    @abstractmethod
    def unlock(self, duration_seconds: float = 5.0) -> bool:
        """
        Unlock the door for a designated duration, then automatically relock.
        
        Args:
            duration_seconds (float): Duration in seconds to stay unlocked before auto-relocking.
            
        Returns:
            bool: True if unlock signal successfully applied, False otherwise.
        """
        pass

    @abstractmethod
    def lock(self) -> bool:
        """
        Immediately lock the door actuator.
        
        Returns:
            bool: True if lock successfully engaged.
        """
        pass

    @abstractmethod
    def get_state(self) -> LockState:
        """
        Retrieve the current hardware/mock lock state.
        
        Returns:
            LockState: Current state (LOCKED or UNLOCKED).
        """
        pass

    def is_locked(self) -> bool:
        """Convenience method checking if door is locked."""
        return self.get_state() == LockState.LOCKED

