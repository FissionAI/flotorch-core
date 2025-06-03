from urllib.parse import urlparse
from flotorch_core.storage.storage import StorageProvider
from flotorch_core.storage.s3_storage import S3StorageProvider
from flotorch_core.storage.local_storage import LocalStorageProvider
from flotorch_core.storage.url_storage import UrlStorageProvider


class StorageProviderFactory:
    """
    Factory to create storage providers based on the URI scheme.
    """
    @staticmethod
    def create_storage_provider(uri: str) -> StorageProvider:
        parsed = urlparse(uri)
        if parsed.scheme == "s3":
            bucket = parsed.netloc
            return S3StorageProvider(bucket)
        elif parsed.scheme in ["http", "https"]:
            return UrlStorageProvider()
        elif parsed.scheme == "" or parsed.scheme == "file":
            return LocalStorageProvider()
        else:
            raise ValueError(f"Unsupported storage scheme: {parsed.scheme}")
