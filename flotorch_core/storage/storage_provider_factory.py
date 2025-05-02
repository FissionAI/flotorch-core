from urllib.parse import urlparse
from flotorch_core.storage.storage import StorageProvider
from flotorch_core.storage.s3_storage import S3StorageProvider
from flotorch_core.storage.local_storage import LocalStorageProvider

class StorageProviderFactory:
    """
    Factory to create storage providers based on the URI scheme.
    """
    @staticmethod
    def create_storage_provider(uri: str) -> StorageProvider:
        """
        Create a storage provider based on the URI scheme.
        Args:
            uri (str): The URI to parse.
        Returns:
            StorageProvider: The storage provider instance."""
        parsed = urlparse(uri)
        if parsed.scheme == "s3":
            bucket = parsed.netloc
            return S3StorageProvider(bucket)
        elif parsed.scheme == "" or parsed.scheme == "file":
            return LocalStorageProvider()
        else:
            raise ValueError(f"Unsupported storage scheme: {parsed.scheme}")
