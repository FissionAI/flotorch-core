import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError
from flotorch_core.storage.db.dynamodb import DynamoDB

@pytest.fixture
def mock_dynamodb():
    """Fixture to create a mocked DynamoDB instance"""
    with patch('boto3.resource') as mock_boto3:
        # Create mock table
        mock_table = Mock()
        mock_boto3.return_value.Table.return_value = mock_table
        
        # Initialize DynamoDB with mocked resources
        db = DynamoDB(table_name='test_table')
        db.table = mock_table
        return db, mock_table

class TestDynamoDB:
    def test_init(self):
        """Test DynamoDB initialization"""
        with patch('boto3.resource') as mock_boto3:
            db = DynamoDB('test_table')
            mock_boto3.assert_called_once_with('dynamodb', region_name='us-east-1')
            mock_boto3.return_value.Table.assert_called_once_with('test_table')

    def test_write_success(self, mock_dynamodb):
        """Test successful write operation"""
        db, mock_table = mock_dynamodb
        test_item = {'id': '1', 'data': 'test'}
        
        result = db.write(test_item)
        
        mock_table.put_item.assert_called_once_with(Item=test_item)
        assert result is True

    def test_write_failure(self, mock_dynamodb):
        """Test write operation failure"""
        db, mock_table = mock_dynamodb
        test_item = {'id': '1', 'data': 'test'}
        
        # Simulate DynamoDB error
        mock_table.put_item.side_effect = ClientError(
            error_response={'Error': {'Message': 'Test error'}},
            operation_name='PutItem'
        )
        
        result = db.write(test_item)
        assert result is False

    def test_read_success(self, mock_dynamodb):
        """Test successful read operation"""
        db, mock_table = mock_dynamodb
        test_key = {'id': '1'}
        expected_item = {'id': '1', 'data': 'test'}
        
        mock_table.get_item.return_value = {'Item': expected_item}
        
        result = db.read(test_key)
        
        mock_table.get_item.assert_called_once_with(Key=test_key)
        assert result == expected_item

    def test_read_not_found(self, mock_dynamodb):
        """Test read operation when item not found"""
        db, mock_table = mock_dynamodb
        test_key = {'id': '1'}
        
        mock_table.get_item.return_value = {}
        
        result = db.read(test_key)
        
        mock_table.get_item.assert_called_once_with(Key=test_key)
        assert result is None

    def test_read_failure(self, mock_dynamodb):
        """Test read operation failure"""
        db, mock_table = mock_dynamodb
        test_key = {'id': '1'}
        
        mock_table.get_item.side_effect = ClientError(
            error_response={'Error': {'Message': 'Test error'}},
            operation_name='GetItem'
        )
        
        result = db.read(test_key)
        assert result is None

    def test_bulk_write_success(self, mock_dynamodb):
        """Test successful bulk write operation"""
        db, mock_table = mock_dynamodb
        test_items = [
            {'id': '1', 'data': 'test1'},
            {'id': '2', 'data': 'test2'}
        ]
        
        # Create a mock batch writer
        mock_batch_writer = Mock()
        
        # Configure the context manager mock properly
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_batch_writer)
        mock_context_manager.__exit__ = Mock(return_value=None)
        
        # Set up the mock table to return our configured context manager
        mock_table.batch_writer.return_value = mock_context_manager
        
        result = db.bulk_write(test_items)
        
        # Verify the batch writer was used correctly
        assert mock_batch_writer.put_item.call_count == 2
        mock_batch_writer.put_item.assert_any_call(Item=test_items[0])
        mock_batch_writer.put_item.assert_any_call(Item=test_items[1])
        assert result is True


    def test_update_success(self, mock_dynamodb):
        """Test successful update operation"""
        db, mock_table = mock_dynamodb
        test_key = {'id': '1'}
        test_data = {'status': 'completed', 'count': 5}
        
        result = db.update(test_key, test_data)
        
        mock_table.update_item.assert_called_once_with(
            Key=test_key,
            UpdateExpression="SET status = :status, count = :count",
            ExpressionAttributeValues={':status': 'completed', ':count': 5},
            ReturnValues="UPDATED_NEW"
        )
        assert result is True

    def test_update_failure(self, mock_dynamodb):
        """Test update operation failure"""
        db, mock_table = mock_dynamodb
        test_key = {'id': '1'}
        test_data = {'status': 'completed'}
        
        mock_table.update_item.side_effect = ClientError(
            error_response={'Error': {'Message': 'Test error'}},
            operation_name='UpdateItem'
        )
        
        result = db.update(test_key, test_data)
        assert result is False


def test_bulk_write_failure(mock_dynamodb):
    db, mock_table = mock_dynamodb

    mock_table.batch_writer.side_effect = ClientError(
        error_response={'Error': {'Message': 'Bulk write error'}},
        operation_name='BatchWriteItem'
    )

    with pytest.raises(ClientError):
        db.bulk_write([{'id': '1'}])

