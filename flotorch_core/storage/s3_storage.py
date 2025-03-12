import logging
import os
from typing import Generator
from urllib.parse import urlparse
import boto3
from .storage import StorageProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3StorageProvider(StorageProvider):
    """
    S3 storage provider
    """

    def __init__(self, bucket: str, s3_client = boto3.client('s3')):

        """
        Initializes the S3Storage class with the specified S3 bucket.
        Args:
            bucket (str): The name of the S3 bucket to interact with.
        Attributes:
            bucket (str): The name of the S3 bucket.
            s3_client (boto3.client): The boto3 client for interacting with S3.
        """

        super().__init__()
        self.bucket = bucket
        self.s3_client = s3_client

    def get_path(self, uri: str) -> str:
        parsed = urlparse(uri)
        return parsed.path.lstrip("/")

    def write(self, path: str, data: bytes) -> None:
        """
        Writes data to the specified path in the S3 bucket.
        Args:
            path (str): The path to write the data to in the S3 bucket.
            data (bytes): The data to write to the S3 bucket.
        """
        logger.info(f'Writing data to S3 storage: {data}')
        if not path.endswith("/"):
            key = path
        else:
            key = path + "tmp.data"
        self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def read(self, path: str) -> Generator[bytes, None, None]:
        """
        Reads data from the specified path in the S3 bucket.
        Args:
            path (str): The path to read the data from in the S3 bucket.
        Returns:
            Generator[bytes, None, None]: A generator that yields the data read from the S3 bucket.
        """
        logger.info('Reading data from S3 storage')
        if self._is_directory(path):
            yield from self._read_directory(path)
        else:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=path)
            yield response['Body'].read()

    def _is_directory(self, path: str) -> bool:
        """
        Determines if the given S3 path is a directory by checking if multiple files exist under the prefix.

        Args:
            path (str): The S3 path.

        Returns:
            bool: True if the path is a directory, False otherwise.
        """
        if not path.endswith("/"):  # Ensure the path ends with a slash to treat it as a directory
            path += "/"

        response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=path, MaxKeys=1)
        return "Contents" in response  # If "Contents" exists, it's a directory

    def _read_directory(self, path: str) -> Generator[bytes, None, None]:
        """
        Reads all files in the specified S3 directory.

        Args:
            path (str): The S3 directory path.

        Returns:
            Generator[bytes, None, None]: Yields file contents.
        """
        if not path.endswith("/"):
            path += "/"

        response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=path)

        if "Contents" not in response:
            logger.warning(f"No files found in S3 directory: {path}")
            return

        for obj in response["Contents"]:
            file_key = obj["Key"]
            if file_key.endswith("/"):  # Skip directories
                continue

            logger.info(f"Reading S3 file: {file_key}")
            file_response = self.s3_client.get_object(Bucket=self.bucket, Key=file_key)
            yield file_response['Body'].read()