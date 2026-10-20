from functools import cache

from common.storage import S3Storage

from backend.config import settings


@cache
def get_storage() -> S3Storage:
    return S3Storage(settings.s3)
