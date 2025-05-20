import boto3
from flotorch_core.storage.db.db_storage import DBStorage
from flotorch_core.logger.global_logger import get_logger
from flotorch_core.utils.db_utils import DBUtils
from botocore.exceptions import ClientError
from typing import List, Dict, Any, Optional

logger = get_logger()

class DynamoDB(DBStorage):
    def __init__(self, table_name, region_name='us-east-1'):
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table = self.dynamodb.Table(table_name)
        self.primary_key_fields = [key['AttributeName'] for key in self.table.key_schema]

    def write(self, item: dict):
        try:
            serialized_data = DBUtils.serialize_data(item)
            self.table.put_item(Item=serialized_data)
            return True
        except ClientError as e:
            logger.error(f"Error writing to DynamoDB: {e}")
            return False

    def read(self, keys: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        items = []
        try:
            if not keys:
                logger.info("No key provided, retrieving all items.")
                response = self.table.scan()
                items.extend(response.get('Items', []))

                # Handle pagination if more results exist
                while 'LastEvaluatedKey' in response:
                    response = self.table.scan(
                        ExclusiveStartKey=response['LastEvaluatedKey']
                    )
                    items.extend(response.get('Items', []))
                return items

            # If keys are provided, retrieved based on it
            if set(keys.keys()) == set(self.primary_key_fields):
                response = self.table.get_item(Key=keys)
                item = response.get('Item', None)
                return [item] if item else []

            # Fallback to scan with filters using pagination
            filter_expression_parts = []
            expression_values = {}
            expression_names = {}

            for k, v in keys.items():
                placeholder_key = f"#k_{k}" 
                placeholder_value = f":v_{k}"
                filter_expression_parts.append(f"{placeholder_key} = {placeholder_value}")
                expression_names[placeholder_key] = k
                expression_values[placeholder_value] = v
            
            filter_expression = " AND ".join(filter_expression_parts)

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
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
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
        
    def delete(self, key: Dict[str, Any]) -> bool:
        try:
            response = self.table.delete_item(
                Key=key,
            )
            logger.info(f"Item with key {key} deleted successfully.")
            return True
        except ClientError as e:
            logger.error(f"Error deleting item with key {key} from DynamoDB: {e}")
            return False        