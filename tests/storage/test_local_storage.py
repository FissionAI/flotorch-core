import pytest
import os
from unittest.mock import patch, mock_open
from flotorch_core.storage.local_storage import LocalStorageProvider

@pytest.fixture
def local_storage():
    return LocalStorageProvider()

def test_get_path(local_storage):
    """Test that get_path correctly extracts local file path."""
    uri = "file:///tmp/testfile.txt"
    assert local_storage.get_path(uri) == "/tmp/testfile.txt"

@patch("builtins.open", new_callable=mock_open)
def test_write(mock_file, local_storage):
    """Test that write correctly writes data to a local file."""
    path = "/tmp/testfile.txt"
    data = b"Hello, World!"
    
    local_storage.write(path, data)
    
    mock_file.assert_called_once_with(path, "wb")
    mock_file().write.assert_called_once_with(data)

@patch("os.path.isdir", return_value=False)
@patch("builtins.open", new_callable=mock_open, read_data=b"Hello, World!")
def test_read(mock_file, mock_isdir, local_storage):
    """Test that read correctly reads data from a local file."""
    path = "/tmp/testfile.txt"
    result = list(local_storage.read(path))  # Convert generator to list
    
    assert result == [b"Hello, World!"]
    mock_file.assert_called_once_with(path, "rb")

@patch("os.listdir", return_value=["file1.txt", "file2.txt"])
@patch("os.path.isfile", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data=b"File content")
def test_read_directory(mock_file, mock_isfile, mock_listdir, local_storage):
    path = "/tmp/testdir"
    result = list(local_storage._read_directory(path))
    assert result == [b"File content", b"File content"]
    assert mock_file.call_count == 2
    mock_listdir.assert_called_once_with(path)
    mock_isfile.assert_any_call(os.path.join(path, "file1.txt"))
    mock_isfile.assert_any_call(os.path.join(path, "file2.txt"))

