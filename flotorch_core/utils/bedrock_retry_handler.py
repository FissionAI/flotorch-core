from typing import Dict
from flotorch_core.utils.boto_retry_handler import BotoRetryHandler, RetryParams


class BedRockRetryHander(BotoRetryHandler):
    """Retry handler for Bedrock service."""
    @property
    def retry_params(self) -> RetryParams:
        return RetryParams(
            max_retries=5,
            retry_delay=2,
            backoff_factor=2
        )
    
    @property
    def retryable_errors(self):
        return {
            "ThrottlingException",
            "ServiceQuotaExceededException",
            "ModelTimeoutException"
        }