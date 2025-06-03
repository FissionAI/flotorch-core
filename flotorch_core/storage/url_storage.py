import logging
import traceback
import requests
from .storage import StorageProvider
from typing import Generator, Any, List, Union, Optional, Dict
from pydantic import BaseModel, Field, ValidationError, parse_obj_as


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UrlStorageProvider(StorageProvider):
    """
    Url storage provider
    """
    def __init__(self):
        super().__init__()

    def get_path(self, uri: str) -> str:
        """
        For HTTP, the entire URI is essentially the "path" to the resource.
        """
        return uri

    def write(self, path: str, data: Union[Dict[str, Any], List[Any], Any], headers: Optional[Dict[str, str]] = None) -> None:
        """
        Sends data as a JSON payload to the specified API path with optional custom headers.

        Args:
            path (str): The API endpoint URL.
            data (Union[Dict[str, Any], List[Any], Any]): The data to send.
                                                            This can be a dictionary, list,
                                                            or any other JSON-serializable Python object.
            headers (Optional[Dict[str, str]]): Optional dictionary of HTTP headers to send with the request.
                                                 If not provided, 'Content-Type: application/json' will be
                                                 automatically set by 'requests' due to the 'json' parameter.
                                                 If provided, these headers will be used, and 'Content-Type'
                                                 will be added if not already present.
        """
        try:
            logger.info(f"Sending data to API URL: {path} with payload: {data}")
            response = requests.post(path, json=data, headers=headers)

            if response.status_code == 200:
                logger.info(f"Successfully sent data to API URL: {path} with status code: {response.status_code}")
            elif response.status_code == 400:
                error_message = f"API Error (400 Bad Request) for {path}: Invalid payload. Response: {response.text}"
                logger.error(error_message)
                raise requests.exceptions.RequestException(error_message)
            elif response.status_code == 404:
                error_message = f"API Error (404 Not Found) for {path}: Invalid project/experiment UUIDs. Response: {response.text}"
                logger.error(error_message)
                raise requests.exceptions.RequestException(error_message)
            else:
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            traceback.print_exc()
            logger.error(f"Error sending data to API URL {path}: {e}")
            raise
        except Exception as e:
            traceback.print_exc()
            logger.error(f"An unexpected error occurred: {e}")
            raise

    def read(self, path: str, headers: Optional[Dict[str, str]] = None) -> Generator[bytes, None, None]:
        """
        Reads data from the specified HTTP(S) URL.
        Args:
            path (str): The URL to read the data from.
            headers (Optional[Dict[str, str]]): Optional dictionary of HTTP headers to send with the request.
        Returns:
            Generator[bytes, None, None]: A generator that yields the data read.
        """
        logger.info(f"Attempting to read from URL: {path}")
        try:
            response = requests.get(path, headers=headers)
            response.raise_for_status()
            yield response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error reading from HTTP(S) path {path}: {e}")
            raise