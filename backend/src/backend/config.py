from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # S3 / MinIO
    s3_endpoint_url: str = "http://localhost:9000"
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin"
    minio_bucket: str = "frames"

    # RabbitMQ
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    face_capture_queue: str = "do_face_capture"

    # Video processing
    target_fps: float = 10.0
    max_frame_height: int = 480
    jpeg_quality: int = 85


settings = Settings()
