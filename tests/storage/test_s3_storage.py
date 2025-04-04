import pytest
from unittest.mock import Mock, patch
from flotorch_core.storage.s3_storage import S3StorageProvider

@pytest.fixture
def mock_s3_client():
    return Mock()

@pytest.fixture
def s3_storage(mock_s3_client):
    return S3StorageProvider(bucket="test-bucket", s3_client=mock_s3_client)

def test_get_path(s3_storage):
    """Test that get_path correctly extracts the S3 key."""
    uri = "s3://test-bucket/path/to/file.txt"
    assert s3_storage.get_path(uri) == "path/to/file.txt"

def test_write(s3_storage, mock_s3_client):
    """Test that write correctly uploads data to S3."""
    path = "path/to/file.txt"
    data = b"Hello, S3!"
    
    s3_storage.write(path, data)
    
    mock_s3_client.put_object.assert_called_once_with(
        Bucket="test-bucket", Key=path, Body=data
    )

@patch("flotorch_core.storage.s3_storage.S3StorageProvider._is_directory", return_value=False)
def test_read(mock_is_directory, s3_storage, mock_s3_client):
    """Test that read fetches an S3 object correctly."""
    path = "path/to/file.txt"
    mock_s3_client.get_object.return_value = {"Body": Mock(read=Mock(return_value=b"File content"))}
    
    result = list(s3_storage.read(path))  # Convert generator to list
    
    assert result == [b"File content"]
    mock_s3_client.get_object.assert_called_once_with(Bucket="test-bucket", Key=path)

@patch("flotorch_core.storage.s3_storage.S3StorageProvider._is_directory", return_value=True)
@patch("flotorch_core.storage.s3_storage.S3StorageProvider._read_directory")
def test_read_directory(mock_read_directory, mock_is_directory, s3_storage):
    """Test that read delegates to _read_directory if the path is a directory."""
    path = "path/to/directory/"
    
    s3_storage.read(path)
    
    mock_read_directory.assert_called_once_with(path)

def test_is_directory(s3_storage, mock_s3_client):
    """Test that _is_directory correctly detects S3 directories."""
    path = "folder/"
    mock_s3_client.list_objects_v2.return_value = {"Contents": [{"Key": "folder/file.txt"}]}
    
    assert s3_storage._is_directory(path) is True

def test_read_directory(s3_storage, mock_s3_client):
    """Test that _read_directory correctly retrieves multiple files."""
    path = "folder/"
    mock_s3_client.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "folder/file1.txt"},
            {"Key": "folder/file2.txt"}
        ]
    }
    mock_s3_client.get_object.side_effect = [
        {"Body": Mock(read=Mock(return_value=b"Content 1"))},
        {"Body": Mock(read=Mock(return_value=b"Content 2"))},
    ]

    result = list(s3_storage._read_directory(path))  # Convert generator to list
    
    assert result == [b"Content 1", b"Content 2"]
    assert mock_s3_client.get_object.call_count == 2
