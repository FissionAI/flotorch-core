import json
from flotorch_core.storage.storage import StorageProvider
from typing import List

"""
This class is responsible for reading the JSON data from the storage.
"""
class JSONReader:
    """
    Initializes the JSONReader object.
    :param storage_provider: The storage provider.
    """

    def __init__(self, storage_provider: StorageProvider):
        self.storage_provider = storage_provider

    """
    Reads the JSON data from the storage.
    :param path: The path of the JSON file.
    :return: The JSON data.
    """
    def read(self, path:str) -> dict:
        data = "".join(chunk.decode("utf-8") for chunk in self.storage_provider.read(path))
        return json.loads(data)
    
    """
    Reads the JSON data from the storage and converts it to a model object.
    :param path: The path of the JSON file.
    :param model_class: The class of the model to convert the JSON data to.
    :return: The model object.
    """
    def read_as_model(self, path: str, model_class: type) -> List[object]:
        data = self.read(path)

        if isinstance(data, list):
            return [model_class(**item) for item in data]
        
        return [model_class(**data)]