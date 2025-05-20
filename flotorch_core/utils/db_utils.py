from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import datetime

class DBUtils:
    """
    Utility class for database operations
    """
    @staticmethod
    def prepare_item_for_write(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Prepares a dictionary item for writing to PostgreSQL.
        Manually converts dictionary values to JSON strings.

        Args:
            item: The dictionary to prepare.

        Returns:
            A new dictionary with dict values converted to JSON strings,
            or None if a serialization error occurs.
        """
        item_prepared = {}
        try:
            for key, value in item.items():
                if isinstance(value, dict):
                    # Convert dictionaries to JSON strings
                    item_prepared[key] = json.dumps(value)
                else:
                    # Keep other types as they are
                    item_prepared[key] = value
            return item_prepared
        except TypeError as e:
            logger.error(f"Error during JSON serialization: {e}")
            return None
    
    @staticmethod
    def serialize_data(obj: Any) -> Any:
        """
        Recursively convert float to Decimal and serialize datetime to string.
        """
        if isinstance(obj, list):
            return [DBUtils.serialize_data(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: DBUtils.serialize_data(v) for k, v in obj.items()}
        elif isinstance(obj, float):
            return Decimal(str(obj))  # Convert float to Decimal
        elif isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO 8601 string
        return obj