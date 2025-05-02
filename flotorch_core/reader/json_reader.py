import json
from flotorch_core.storage.storage import StorageProvider
from typing import List


class JSONReader:
    """
    This class is responsible for reading the JSON data from the storage.
    """

    def __init__(self, storage_provider: StorageProvider):
        """
        Initializes the JSONReader object.
        Args:
            storage_provider (StorageProvider): The storage provider to read the JSON data from.
        """
        self.storage_provider = storage_provider

    def read(self, path:str) -> dict:
        """
        Reads the JSON data from the storage.
        Args:
            path (str): The path of the JSON file.
        Returns:
            dict: The JSON data as a dictionary.
        """
        data = "".join(chunk.decode("utf-8") for chunk in self.storage_provider.read(path))
        return json.loads(data)
    
    def read_as_model(self, path: str, model_class: type) -> List[object]:
        """
        Reads the JSON data from the storage and converts it to a model object.
        Args:
            path (str): The path of the JSON file.
            model_class (type): The model class to convert the JSON data to.
        Returns:
            List[object]: A list of model objects.
        """
        data = self.read(path)

        if isinstance(data, list):
            return [model_class(**item) for item in data]
        
        return [model_class(**data)]