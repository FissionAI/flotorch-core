import time
import unittest
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError
from flotorch_core.utils.boto_retry_handler import BotoRetryHandler, RetryParams
from flotorch_core.utils.bedrock_retry_handler import BedRockRetryHander

class TestBotoRetryHandler(unittest.TestCase):
    
    def setUp(self):
        self.retry_handler = BedRockRetryHander()
    
    @patch("time.sleep", return_value=None)  # Patch sleep to speed up tests
    def test_successful_call(self, mock_sleep):
        mock_func = MagicMock(return_value="Success")
        wrapped_func = self.retry_handler(mock_func)
        result = wrapped_func()
        self.assertEqual(result, "Success")
        mock_func.assert_called_once()
    
    @patch("time.sleep", return_value=None)
    def test_retryable_error(self, mock_sleep):
        mock_func = MagicMock()
        error_response = {"Error": {"Code": "ThrottlingException"}}
        mock_func.side_effect = [ClientError(error_response, "TestOp")] * 3 + ["Success"]
        
        wrapped_func = self.retry_handler(mock_func)
        result = wrapped_func()
        
        self.assertEqual(result, "Success")
        self.assertEqual(mock_func.call_count, 4)  # 3 failures + 1 success
    
    @patch("time.sleep", return_value=None)
    def test_max_retries_exceeded(self, mock_sleep):
        mock_func = MagicMock()
        error_response = {"Error": {"Code": "ThrottlingException"}}
        mock_func.side_effect = ClientError(error_response, "TestOp")
        
        wrapped_func = self.retry_handler(mock_func)
        
        with self.assertRaises(ClientError):
            wrapped_func()
        
        self.assertEqual(mock_func.call_count, self.retry_handler.retry_params.max_retries)
    
    @patch("time.sleep", return_value=None)
    def test_non_retryable_error(self, mock_sleep):
        mock_func = MagicMock()
        error_response = {"Error": {"Code": "AccessDeniedException"}}  # Not in retryable_errors
        mock_func.side_effect = ClientError(error_response, "TestOp")
        
        wrapped_func = self.retry_handler(mock_func)
        
        with self.assertRaises(ClientError):
            wrapped_func()
        
        self.assertEqual(mock_func.call_count, 1)  # Should not retry for non-retryable errors
    
    @patch("time.sleep", return_value=None)
    def test_unexpected_exception(self, mock_sleep):
        mock_func = MagicMock(side_effect=ValueError("Unexpected error"))
        wrapped_func = self.retry_handler(mock_func)
        
        with self.assertRaises(ValueError):
            wrapped_func()
        
        self.assertEqual(mock_func.call_count, 1)  # Should not retry for unexpected errors

if __name__ == "__main__":
    unittest.main()
