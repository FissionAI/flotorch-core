from abc import ABC, abstractmethod
import time
from pydantic import BaseModel
import botocore
from flotorch_core.logger.global_logger import get_logger

logger = get_logger()

class RetryParams(BaseModel):
    max_retries: int
    retry_delay: int
    backoff_factor: int


class BotoRetryHandler(ABC):
    """Abstract class for retry handler"""
    
    @property
    @abstractmethod
    def retry_params(self) -> RetryParams:
        pass
    
    @property
    @abstractmethod
    def retryable_errors(self) -> set[str]:
        pass
        
        
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            retries = 0
            retry_params = self.retry_params
            while retries < retry_params.max_retries:
                try:
                    return func(*args, **kwargs)
                except botocore.exceptions.ClientError as e:
                    error_code = e.response['Error']['Code']
                    if error_code in self.retryable_errors:
                        retries += 1
                        logger.error(f"Rate limit error in Bedrock converse (Attempt {retries}/{retry_params.max_retries}): {str(e)}")
                        
                        if retries >= retry_params.max_retries:
                            logger.error("Max retries reached. Could not complete Bedrock converse operation.")
                            raise
                        
                        backoff_time = retry_params.retry_delay * (retry_params.backoff_factor ** (retries - 1))
                        logger.info(f"Retrying in {backoff_time} seconds...")
                        time.sleep(backoff_time)
                    else:
                        # If it's not a rate limit error, raise immediately
                        raise
                except Exception as e:
                    # For any other exception, log and raise immediately
                    logger.error(f"Unexpected error in Bedrock converse: {str(e)}")
                    raise
            
        return wrapper