from typing import Dict
from flotorch_core.utils.boto_retry_handler import BotoRetryHandler, RetryParams


class BedRockRetryHander(BotoRetryHandler):
    """Retry handler for Bedrock service."""
    @property
    def retry_params(self) -> RetryParams:
        """
        Returns the retry parameters for the Bedrock service.
        Returns:
            RetryParams: The retry parameters for the Bedrock service.
        """
        return RetryParams(
            max_retries=5,
            retry_delay=2,
            backoff_factor=2
        )
    
    @property
    def retryable_errors(self):
        """
        Returns a set of retryable errors for the Bedrock service.
        Returns:
            set: A set of retryable errors for the Bedrock service.
        """
        return {
            "ThrottlingException",
            "ServiceQuotaExceededException",
            "ModelTimeoutException"
        }