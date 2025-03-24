import pytest
import boto3
from unittest.mock import Mock, patch
from flotorch_core.storage.storage_provider_factory import StorageProviderFactory
from flotorch_core.storage.local_storage import LocalStorageProvider
from flotorch_core.storage.s3_storage import S3StorageProvider
from unittest.mock import patch, Mock


def test_create_storage_provider_local_path():
    """Test creating storage provider with local file path"""
    # Test data
    local_path = "file:///Users/dheeratallapragada/Downloads/medical_20_qa_small_questions.json"
    
    # Execute
    storage = StorageProviderFactory.create_storage_provider(local_path)
    
    # Assert
    assert isinstance(storage, LocalStorageProvider)
    assert storage.get_path(local_path) == "/Users/dheeratallapragada/Downloads/medical_20_qa_small_questions.json"

def test_create_storage_provider_s3_path():
    """Test creating storage provider with S3 path"""
    # Test data
    s3_path = "s3://flotorch-data-refact/bfa24db3-ab76-47e7-9ff3-9e942a0a2a63/gt_data/gt.json"
    
    # Execute
    storage = StorageProviderFactory.create_storage_provider(s3_path)
    
    # Assert
    assert isinstance(storage, S3StorageProvider)
    assert storage.get_path(s3_path) == "bfa24db3-ab76-47e7-9ff3-9e942a0a2a63/gt_data/gt.json"

def test_create_storage_provider_invalid_path():
    """Test creating storage provider with invalid path"""
    # Test data
    invalid_path = "invalid://path/to/file.json"
    
    # Assert
    with pytest.raises(ValueError) as exc_info:
        StorageProviderFactory.create_storage_provider(invalid_path)
    assert "Unsupported storage scheme: invalid" in str(exc_info.value)

@patch('flotorch_core.storage.s3_storage.boto3.client', autospec=True)
def test_create_storage_provider_s3_with_mock(mock_client):
    """Test creating storage provider with S3 path using mocked boto3"""
    # Test data
    s3_path = "s3://flotorch-data-refact/test/file.json"

    # Create a fresh mock for the S3 client
    mock_s3_client = Mock()
    mock_client.return_value = mock_s3_client

    # Force reload of the module to ensure our mock is used
    import importlib
    import flotorch_core.storage.s3_storage
    importlib.reload(flotorch_core.storage.s3_storage)

    # Execute
    storage = StorageProviderFactory.create_storage_provider(s3_path)

    # Assert
    assert isinstance(storage, S3StorageProvider)
    assert storage.bucket == "flotorch-data-refact"
    assert storage.get_path(s3_path) == "test/file.json"
    
    # Verify the mock was called
    mock_client.assert_called_with('s3')


# Use below if s3_storage.py accepts s3_client = None
# @patch('boto3.client')
# def test_create_storage_provider_s3_with_mock(mock_boto3_client):
#     """Test creating storage provider with S3 path using mocked boto3"""
#     # Test data
#     s3_path = "s3://flotorch-data-refact/test/file.json"

#     # Mock boto3 client
#     mock_s3_client = Mock()
#     mock_boto3_client.return_value = mock_s3_client

#     # Execute
#     storage = StorageProviderFactory.create_storage_provider(s3_path)

#     # Assert
#     assert isinstance(storage, S3StorageProvider)
#     assert storage.bucket == "flotorch-data-refact"
#     assert storage.get_path(s3_path) == "test/file.json"

#     # Verify that boto3.client('s3') was called once
#     mock_boto3_client.assert_called_once_with('s3')



