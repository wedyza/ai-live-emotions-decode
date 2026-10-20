import asyncio
from typing import TYPE_CHECKING

import boto3

from common.settings import S3Settings

if TYPE_CHECKING:
    # Стабы только для разработки (boto3-stubs-lite[s3]), в рантайме не установлены.
    from mypy_boto3_s3 import S3Client


class S3Storage:
    """Тонкая обёртка над boto3. Синхронные методы блокируют; из async-кода вызывайте *_async."""

    _client: "S3Client"
    bucket: str

    def __init__(self, settings: S3Settings) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint_url,
            aws_access_key_id=settings.access_key,
            aws_secret_access_key=settings.secret_key.get_secret_value(),
        )
        self.bucket = settings.bucket

    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        _ = self._client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
        )

    def get(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    async def put_async(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        await asyncio.to_thread(self.put, key, data, content_type)

    async def get_async(self, key: str) -> bytes:
        return await asyncio.to_thread(self.get, key)
