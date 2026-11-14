"""System configuration settings loaded from environment variables or .env file."""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with validation and type safety."""
    
    # Camera settings
    camera_index: int = Field(default=0, description="Webcam device index (0 for default built-in webcam)")
    camera_frame_width: int = Field(default=640, description="Capture frame width in pixels")
    camera_frame_height: int = Field(default=480, description="Capture frame height in pixels")
    camera_fps: int = Field(default=30, description="Target capture frames per second")
    
    # ESP32 settings (Phase 6 preparation)
    esp32_stream_url: str = Field(default="http://192.168.1.100:81/stream", description="ESP32-S3-CAM MJPEG stream URL")
    esp32_timeout_seconds: float = Field(default=5.0, description="HTTP connection timeout in seconds")
    
    # Lock settings
    lock_auto_relock_seconds: float = Field(default=5.0, description="Seconds before unlocked door automatically locks")
    lock_fail_secure: bool = Field(default=True, description="Fail-secure mode: locked by default upon any failure")
    
    # Biometric settings (Phase 3 & 4 preparation)
    face_similarity_threshold: float = Field(default=0.60, description="Cosine similarity threshold for face match")
    palm_similarity_threshold: float = Field(default=0.70, description="Matching threshold for palmprint verification")
    
    # Database
    database_path: str = Field(default="sqlite:///./smart_door_lock.db", description="SQLite connection string")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached instance of application settings."""
    return Settings()

