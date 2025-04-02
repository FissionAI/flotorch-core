
import unittest
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError
from pydantic import ValidationError

from flotorch_core.utils.boto_retry_handler import BotoRetryHandler, RetryParams

class TestRetryParams(unittest.TestCase):
    """Tests for the RetryParams model"""
    
    def test_valid_retry_params(self):
        params = RetryParams(max_retries=3, retry_delay=1, backoff_factor=2)
        self.assertEqual(params.max_retries, 3)
        self.assertEqual(params.retry_delay, 1)
        self.assertEqual(params.backoff_factor, 2)
    
    def test_invalid_retry_params(self):
        with self.assertRaises(ValidationError):
            RetryParams(max_retries="invalid", retry_delay=1, backoff_factor=2)
        
        with self.assertRaises(ValidationError):
            RetryParams(max_retries=3, retry_delay="invalid", backoff_factor=2)
        
        with self.assertRaises(ValidationError):
            RetryParams(max_retries=3, retry_delay=1, backoff_factor="invalid")


class ConcreteBotoRetryHandler(BotoRetryHandler):
    """A concrete implementation of BotoRetryHandler for testing"""
    
    @property
    def retry_params(self) -> RetryParams:
        return RetryParams(max_retries=3, retry_delay=1, backoff_factor=2)
    
    @property
    def retryable_errors(self) -> set[str]:
        return {"ThrottlingException", "RequestLimitExceeded"}


class TestBotoRetryHandler(unittest.TestCase):
    """Tests for the abstract BotoRetryHandler class"""
    
    def setUp(self):
        self.retry_handler = ConcreteBotoRetryHandler()
    
    def test_abstract_class_instantiation(self):
        """Test that direct instantiation of the abstract class raises TypeError"""
        with self.assertRaises(TypeError):
            BotoRetryHandler()
    
    @patch("time.sleep", return_value=None)  # Patch sleep to speed up tests
    def test_successful_call(self, mock_sleep):
        mock_func = MagicMock(return_value="Success")
        wrapped_func = self.retry_handler(mock_func)
        result = wrapped_func()
        self.assertEqual(result, "Success")
        mock_func.assert_called_once()
        mock_sleep.assert_not_called()
    
    @patch("time.sleep", return_value=None)
    def test_retryable_error(self, mock_sleep):
        mock_func = MagicMock()
        error_response = {"Error": {"Code": "ThrottlingException"}}
        mock_func.side_effect = [
            ClientError(error_response, "TestOp"),
            "Success"
        ]
        
        wrapped_func = self.retry_handler(mock_func)
        result = wrapped_func()
        
        self.assertEqual(result, "Success")
        self.assertEqual(mock_func.call_count, 2)  # 1 failure + 1 success
        mock_sleep.assert_called_once_with(1)  # First retry delay
    
    @patch("time.sleep", return_value=None)
    def test_multiple_retries(self, mock_sleep):
        mock_func = MagicMock()
        error_response = {"Error": {"Code": "ThrottlingException"}}
        mock_func.side_effect = [
            ClientError(error_response, "TestOp"),
            ClientError(error_response, "TestOp"),
            "Success"
        ]
        
        wrapped_func = self.retry_handler(mock_func)
        result = wrapped_func()
        
        self.assertEqual(result, "Success")
        self.assertEqual(mock_func.call_count, 3)  # 2 failures + 1 success
        # Check that sleep was called with the correct backoff times
        mock_sleep.assert_any_call(1)  # First retry
        mock_sleep.assert_any_call(2)  # Second retry with backoff
    
    @patch("time.sleep", return_value=None)
    def test_max_retries_exceeded(self, mock_sleep):
        mock_func = MagicMock()
        error_response = {"Error": {"Code": "ThrottlingException"}}
        
        # Create a ClientError object that we'll use consistently
        client_error = ClientError(error_response, "TestOp")
        
        # Set up the mock to always raise the same error
        mock_func.side_effect = client_error
        
        wrapped_func = self.retry_handler(mock_func)
        
        # We expect the function to raise the same ClientError after max retries
        with self.assertRaises(ClientError) as context:
            wrapped_func()
            
        # Verify it's the same error object
        self.assertIs(context.exception, client_error)
        
        # Should try exactly max_retries times
        self.assertEqual(mock_func.call_count, self.retry_handler.retry_params.max_retries)
        
        # Check sleep was called with correct backoff times
        self.assertEqual(mock_sleep.call_count, self.retry_handler.retry_params.max_retries - 1)
        mock_sleep.assert_any_call(1)  # First retry
        mock_sleep.assert_any_call(2)  # Second retry with backoff
        # No need to check for the third call since we're only doing max_retries=3
    
    @patch("time.sleep", return_value=None)
    def test_different_retryable_error(self, mock_sleep):
        mock_func = MagicMock()
        error_response = {"Error": {"Code": "RequestLimitExceeded"}}  # Different retryable error
        mock_func.side_effect = [
            ClientError(error_response, "TestOp"),
            "Success"
        ]
        
        wrapped_func = self.retry_handler(mock_func)
        result = wrapped_func()
        
        self.assertEqual(result, "Success")
        self.assertEqual(mock_func.call_count, 2)
    
    @patch("time.sleep", return_value=None)
    def test_non_retryable_error(self, mock_sleep):
        mock_func = MagicMock()
        error_response = {"Error": {"Code": "AccessDeniedException"}}  # Not in retryable_errors
        mock_func.side_effect = ClientError(error_response, "TestOp")
        
        wrapped_func = self.retry_handler(mock_func)
        
        with self.assertRaises(ClientError):
            wrapped_func()
        
        self.assertEqual(mock_func.call_count, 1)  # Should not retry for non-retryable errors
        mock_sleep.assert_not_called()
    
    @patch("time.sleep", return_value=None)
    def test_unexpected_exception(self, mock_sleep):
        mock_func = MagicMock(side_effect=ValueError("Unexpected error"))
        wrapped_func = self.retry_handler(mock_func)
        
        with self.assertRaises(ValueError):
            wrapped_func()
        
        self.assertEqual(mock_func.call_count, 1)  # Should not retry for unexpected errors
        mock_sleep.assert_not_called()
    
    @patch("flotorch_core.utils.boto_retry_handler.logger")  # Patch logger directly
    @patch("time.sleep", return_value=None)
    def test_logging(self, mock_sleep, mock_logger):
        # Mock function that raises a retryable error first
        mock_func = MagicMock()
        error_response = {"Error": {"Code": "ThrottlingException"}}
        mock_func.side_effect = [
            ClientError(error_response, "TestOp"),
            "Success"
        ]

        # Call wrapped function
        wrapped_func = self.retry_handler(mock_func)
        result = wrapped_func()

        # Ensure function succeeded
        self.assertEqual(result, "Success")

        # Verify that logger.error was called
        mock_logger.error.assert_called()  
        mock_logger.info.assert_any_call("Retrying in 1 seconds...")

if __name__ == "__main__":
    unittest.main()
