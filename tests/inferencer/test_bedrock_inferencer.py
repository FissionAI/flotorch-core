import unittest
from unittest.mock import MagicMock, patch
import pytest
import random

from flotorch_core.inferencer.inferencer import BaseInferencer
from flotorch_core.inferencer.bedrock_inferencer import BedrockInferencer


class TestBedrockInferencer(unittest.TestCase):
    def setUp(self):
        # Mock the boto3 client
        self.boto3_client_patch = patch('boto3.client')
        self.mock_boto3_client = self.boto3_client_patch.start()
        
        # Mock client instance
        self.mock_client = MagicMock()
        self.mock_boto3_client.return_value = self.mock_client
        
        # Create inferencer instance
        self.inferencer = BedrockInferencer(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            region="us-east-1",
            n_shot_prompts=2,
            temperature=0.5,
            n_shot_prompt_guide_obj={
                "system_prompt": "Custom system prompt",
                "user_prompt": "Custom user prompt",
                "examples": [
                    {"question": "Question 1", "answer": "Answer 1"},
                    {"question": "Question 2", "answer": "Answer 2"},
                    {"question": "Question 3", "answer": "Answer 3"}
                ]
            }
        )
        
    def tearDown(self):
        self.boto3_client_patch.stop()

    def test_initialization(self):
        """Test proper initialization of the BedrockInferencer."""
        self.assertEqual(self.inferencer.model_id, "anthropic.claude-3-sonnet-20240229-v1:0")
        # Region is not stored as an attribute in the class
        self.assertEqual(self.inferencer.n_shot_prompts, 2)
        self.assertEqual(self.inferencer.temperature, 0.5)
        self.assertIsNotNone(self.inferencer.n_shot_prompt_guide_obj)
        self.assertEqual(self.inferencer.client, self.mock_client)
        
    def test_prepare_conversation(self):
        """Test formatting messages into conversation format."""
        result = self.inferencer._prepare_conversation(role="user", message="Test message")
        expected = {"role": "user", "content": [{"text": "Test message"}]}
        self.assertEqual(result, expected)
        
    def test_format_context_empty(self):
        """Test formatting empty context."""
        result = self.inferencer.format_context([])
        self.assertEqual(result, "")
        
    def test_format_context(self):
        """Test formatting context with documents."""
        context = [
            {"text": "Document 1 content"},
            {"text": "Document 2 content"}
        ]
        result = self.inferencer.format_context(context)
        expected = "Context 1:\nDocument 1 content\nContext 2:\nDocument 2 content"
        self.assertEqual(result, expected)
        
    def test_extract_response(self):
        """Test extracting text from Bedrock response."""
        mock_response = {
            "output": {
                "message": {
                    "content": [{"text": "Generated response"}]
                }
            }
        }
        result = self.inferencer._extract_response(mock_response)
        self.assertEqual(result, "Generated response")
        
    @patch('random.sample')
    def test_generate_prompt_with_examples(self, mock_random_sample):
        """Test prompt generation with examples."""
        # Set the random.sample to return predictable results
        mock_random_sample.return_value = [
            {"question": "Question 1", "answer": "Answer 1"},
            {"question": "Question 2", "answer": "Answer 2"}
        ]
        
        system_prompt, messages = self.inferencer.generate_prompt("What is the capital of France?")
        
        self.assertEqual(system_prompt, "Custom system prompt")
        # Updated expectation: user_prompt + 2 examples (Q&A pairs = 4 messages) + user query = 6
        self.assertEqual(len(messages), 6)
        self.assertEqual(messages[0]["content"][0]["text"], "Custom user prompt")
        self.assertEqual(messages[1]["content"][0]["text"], "Question 1")
        self.assertEqual(messages[2]["content"][0]["text"], "Answer 1")
        self.assertEqual(messages[3]["content"][0]["text"], "Question 2")
        self.assertEqual(messages[4]["content"][0]["text"], "Answer 2")
        self.assertEqual(messages[5]["content"][0]["text"], "What is the capital of France?")
        
    def test_generate_prompt_with_context(self):
        """Test prompt generation with context."""
        context = [{"text": "France is a country in Europe. Its capital is Paris."}]
        system_prompt, messages = self.inferencer.generate_prompt("What is the capital of France?", context)
        
        self.assertEqual(system_prompt, "Custom system prompt")
        self.assertTrue(len(messages) > 1)
        self.assertEqual(messages[0]["content"][0]["text"], "Context 1:\nFrance is a country in Europe. Its capital is Paris.")
        
    def test_generate_text_skip_system_param(self):
        """Test generate_text with model that requires skipping system param."""
        self.inferencer.model_id = "amazon.titan-text-express-v1"
        
        # Mock the response
        mock_response = {
            "output": {
                "message": {
                    "content": [{"text": "Paris is the capital of France."}]
                }
            },
            "usage": {"input_tokens": 50, "output_tokens": 10},
            "metrics": {"latency": 0.5}
        }
        self.mock_client.converse.return_value = mock_response
        
        metadata, response = self.inferencer.generate_text("What is the capital of France?")
        
        # Verify the client was called with correct parameters
        call_args = self.mock_client.converse.call_args[1]
        self.assertEqual(call_args["modelId"], "amazon.titan-text-express-v1")
        self.assertTrue("system" not in call_args)
        self.assertTrue(len(call_args["messages"]) > 0)
        
        # Verify response processing
        self.assertEqual(response, "Paris is the capital of France.")
        self.assertEqual(metadata["input_tokens"], 50)
        self.assertEqual(metadata["output_tokens"], 10)
        self.assertEqual(metadata["latency"], 0.5)
        
    def test_generate_text_with_system_param(self):
        """Test generate_text with model that uses system param."""
        # Mock the response
        mock_response = {
            "output": {
                "message": {
                    "content": [{"text": "Paris is the capital of France."}]
                }
            }
        }
        self.mock_client.converse.return_value = mock_response
        
        metadata, response = self.inferencer.generate_text("What is the capital of France?")
        
        # Verify the client was called with correct parameters
        call_args = self.mock_client.converse.call_args[1]
        self.assertEqual(call_args["modelId"], "anthropic.claude-3-sonnet-20240229-v1:0")
        self.assertTrue("system" in call_args)
        self.assertEqual(call_args["system"][0]["text"], "Custom system prompt")
        
        # Verify response processing
        self.assertEqual(response, "Paris is the capital of France.")
        
    def test_error_handling(self):
        """Test error handling in generate_text."""
        self.mock_client.converse.side_effect = Exception("API Error")
        
        with self.assertRaises(Exception):
            self.inferencer.generate_text("What is the capital of France?")
        
    def test_generate_prompt_negative_n_shot(self):
        """Test generate_prompt with negative n_shot_prompts."""
        self.inferencer.n_shot_prompts = -1
        
        with self.assertRaises(ValueError):
            self.inferencer.generate_prompt("Test query")
            
    def test_generate_prompt_default_system_prompt(self):
        """Test generate_prompt with no system prompt in guide object."""
        self.inferencer.n_shot_prompt_guide_obj = {"examples": []}
        
        system_prompt, _ = self.inferencer.generate_prompt("Test query")
        self.assertEqual(system_prompt, "You are a helpful assistant. Use the provided context to answer questions accurately. If you cannot find the answer in the context, say so")
        
    def test_generate_prompt_no_guide_obj(self):
        """Test generate_prompt with no guide object."""
        self.inferencer.n_shot_prompt_guide_obj = None
        
        system_prompt, messages = self.inferencer.generate_prompt("Test query")
        self.assertEqual(system_prompt, "You are a helpful assistant. Use the provided context to answer questions accurately. If you cannot find the answer in the context, say so")
        self.assertEqual(len(messages), 1)  # Only the user query
        
    def test_generate_prompt_example_format(self):
        """Test generate_prompt with different example formats."""
        self.inferencer.n_shot_prompt_guide_obj = {
            "examples": [
                {"example": "This is a single example"}
            ]
        }
        self.inferencer.n_shot_prompts = 1
        
        _, messages = self.inferencer.generate_prompt("Test query")
        self.assertEqual(messages[0]["content"][0]["text"], "This is a single example")


if __name__ == "__main__":
    unittest.main()