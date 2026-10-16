import boto3

from backend.config import settings


class FrameStorage:
    def __init__(self) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.minio_root_user,
            aws_secret_access_key=settings.minio_root_password,
        )
        self.bucket = settings.minio_bucket

    def put_jpeg(self, key: str, data: bytes) -> None:
        self._client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType="image/jpeg"
        )
