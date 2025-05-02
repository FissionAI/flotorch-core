import boto3
from flotorch_core.storage.db.db_storage import DBStorage
from flotorch_core.logger.global_logger import get_logger
from botocore.exceptions import ClientError
from typing import List, Dict, Any

logger = get_logger()

class DynamoDB(DBStorage):
    """
    DynamoDB storage class for storing and retrieving data.
    This class provides methods to write, read, and update items in a DynamoDB table.
    """
    def __init__(self, table_name, region_name='us-east-1'):
        """
        Initialize the DynamoDB storage class.
        Args:
            table_name (str): The name of the DynamoDB table.
            region_name (str): The AWS region where the DynamoDB table is located.
        """
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table = self.dynamodb.Table(table_name)
        self.primary_key_fields = [key['AttributeName'] for key in self.table.key_schema]

    def write(self, item: dict):
        """
        Write a single item to the DynamoDB table.
        Args:
            item (dict): The item to be written to the table.
        Returns:
            bool: True if the item was written successfully, False otherwise.
        """
        try:
            self.table.put_item(Item=item)
            return True
        except ClientError as e:
            logger.error(f"Error writing to DynamoDB: {e}")
            return False

    def read(self, keys: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Read an item from the DynamoDB table.
        Args:
            key (dict): The key of the item to be read from the table.
        Returns:
            dict: The item read from the table, or None if not found.
        """
        try:
            if set(keys.keys()) == set(self.primary_key_fields):
                response = self.table.get_item(Key=keys)
                item = response.get('Item', None)
                return [item] if item else []

            # Fallback to scan with filters using pagination
            filter_expression = " AND ".join(f"#{k} = :{k}" for k in keys)
            expression_values = {f":{k}": v for k, v in keys.items()}
            expression_names = {f"#{k}": k for k in keys}

            items = []
            response = self.table.scan(
                FilterExpression=filter_expression,
                ExpressionAttributeNames=expression_names,
                ExpressionAttributeValues=expression_values
            )
            items.extend(response.get('Items', []))

            # Handle pagination if more results exist
            while 'LastEvaluatedKey' in response:
                response = self.table.scan(
                    FilterExpression=filter_expression,
                    ExpressionAttributeNames=expression_names,
                    ExpressionAttributeValues=expression_values,
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))

            return items
        except ClientError as e:
            logger.error(f"Error reading from DynamoDB: {e}")
            return []
    
    def bulk_write(self, items: list):
        """
        Bulk write items to the DynamoDB table.
        Args:
            items (list): A list of items to be written to the table.
        Returns:
            bool: True if the items were written successfully, False otherwise.
        """
        with self.table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=item)
        return True
    
    def update(self, key: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """
        Update method accepts:
        - `key`: Unique identifier to find the record (e.g., {'id': 123})
        - `data`: Fields to be updated with new values (e.g., {'status': 'completed'})
        Args:
            key (dict): The key of the item to be updated.
            data (dict): The data to update in the item.
        Returns:
            bool: True if the item was updated successfully, False otherwise.
        """
        try:
            # Dynamically construct UpdateExpression and ExpressionAttributeValues
            update_expression = "SET " + ", ".join(f"{k} = :{k}" for k in data.keys())
            expression_values = {f":{k}": v for k, v in data.items()}

            self.table.update_item(
                Key=key,
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values,
                ReturnValues="UPDATED_NEW"
            )
            return True
        except ClientError as e:
            logger.error(f"Error updating DynamoDB: {e}")
            return False
    