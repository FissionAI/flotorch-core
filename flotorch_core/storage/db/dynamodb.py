import boto3
from flotorch_core.storage.db.db_storage import DBStorage
from flotorch_core.logger.global_logger import get_logger
from botocore.exceptions import ClientError
from typing import List, Dict, Any

logger = get_logger()

class DynamoDB(DBStorage):
    def __init__(self, table_name, region_name='us-east-1'):
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table = self.dynamodb.Table(table_name)
        self.primary_key_fields = [key['AttributeName'] for key in self.table.key_schema]

    def write(self, item: dict):
        try:
            self.table.put_item(Item=item)
            return True
        except ClientError as e:
            logger.error(f"Error writing to DynamoDB: {e}")
            return False

    def read(self, keys: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        with self.table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=item)
        return True
    
    def update(self, key: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """
        Update method accepts:
        - `key`: Unique identifier to find the record (e.g., {'id': 123})
        - `data`: Fields to be updated with new values (e.g., {'status': 'completed'})
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
    