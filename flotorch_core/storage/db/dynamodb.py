import boto3
from flotorch_core.storage.db.db_storage import DBStorage
from botocore.exceptions import ClientError
from typing import List, Dict, Any

class DynamoDB(DBStorage):
    def __init__(self, table_name, region_name='us-east-1'):
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table = self.dynamodb.Table(table_name)

    def write(self, item: dict):
        try:
            self.table.put_item(Item=item)
            return True
        except ClientError as e:
            print(f"Error writing to DynamoDB: {e}")
            return False

    def read(self, key) -> dict:
        try:
            response = self.table.get_item(Key=key)
            return response.get('Item', None)
        except ClientError as e:
            print(f"Error reading from DynamoDB: {e}")
            return None
    
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
            print(f"Error updating DynamoDB: {e}")
            return False
    
    def query(self, key_condition_expression, expression_attribute_values=None, filter_expression=None, index_name=None):
        """
        Query items from DynamoDB table based on key condition expression
        
        Args:
            key_condition_expression (str): Key condition expression for query
            expression_attribute_values (dict, optional): Values for the expression
            filter_expression (str, optional): Filter expression to apply
            index_name (str, optional): Name of the index to query
            
        Returns:
            List of items matching the query
        """
        try:
            params = {
                'KeyConditionExpression': key_condition_expression,
            }
            
            if expression_attribute_values:
                params['ExpressionAttributeValues'] = expression_attribute_values
                
            if filter_expression:
                params['FilterExpression'] = filter_expression
                
            if index_name:
                params['IndexName'] = index_name
                
            response = self.table.query(**params)
            items = response.get('Items', [])
            
            # Handle pagination if there are more results
            while 'LastEvaluatedKey' in response:
                params['ExclusiveStartKey'] = response['LastEvaluatedKey']
                response = self.table.query(**params)
                items.extend(response.get('Items', []))
                
            return items
        except ClientError as e:
            print(f"Error querying DynamoDB: {e}")
            return []