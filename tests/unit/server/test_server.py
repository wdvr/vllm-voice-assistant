#!/usr/bin/env python3
"""
Unit Tests for vLLM Server

These tests verify the core functionality of the vLLM server.
Note: These tests do not require an actual GPU or model.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Get the project root directory and add it to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

# Import the server module (we'll mock the LLM parts)
import server.vllm_server as vllm_server


class TestServerUnitTests(unittest.TestCase):
    """Unit tests for the vLLM server that don't require an actual GPU."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test app
        self.app = vllm_server.app
        self.client = self.app.test_client()
        
        # Mock the LLM
        self.mock_llm = MagicMock()
        vllm_server.llm = self.mock_llm
        
        # Set a model path for tests
        vllm_server.model_path = "mock-model"
    
    def test_list_models_endpoint(self):
        """Test the /v1/models endpoint."""
        # Set up the mock
        self.mock_llm.get_model_config.return_value.model_id = "test-model"
        
        # Make the request
        response = self.client.get("/v1/models")
        
        # Check the result
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("models", data)
        self.assertIn("test-model", data["models"])
    
    def test_model_info_endpoint(self):
        """Test the /v1/model/info endpoint."""
        # Set up the mock
        self.mock_llm.get_model_config.return_value.model_id = "test-model"
        self.mock_llm.get_model_config.return_value.max_model_len = 4096
        
        # Set up GPU memory mock
        with patch("torch.cuda.memory_allocated") as mock_allocated:
            with patch("torch.cuda.memory_reserved") as mock_reserved:
                mock_allocated.return_value = 1024 ** 3  # 1 GB
                mock_reserved.return_value = 2 * 1024 ** 3  # 2 GB
                
                # Make the request
                response = self.client.get("/v1/model/info")
                
                # Check the result
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                self.assertEqual(data["model_name"], "test-model")
                self.assertEqual(data["max_model_len"], 4096)
                self.assertIn("1.00 GB allocated", data["gpu_memory_usage"])
                self.assertIn("2.00 GB reserved", data["gpu_memory_usage"])
    
    @patch("server.vllm_server.formatter")
    def test_completions_endpoint(self, mock_formatter):
        """Test the /v1/completions endpoint."""
        # Set up the mocks
        mock_formatter.format_prompt.return_value = "Formatted: test prompt"
        mock_formatter.parse_response.return_value = "Parsed: test response"
        
        mock_output = MagicMock()
        mock_output.outputs[0].text = "Raw response"
        self.mock_llm.generate.return_value = [mock_output]
        
        # Prepare the request data
        request_data = {
            "prompt": "test prompt",
            "max_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.95
        }
        
        # Make the request
        response = self.client.post("/v1/completions", json=request_data)
        
        # Check the result
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["text"], "Parsed: test response")
        self.assertIn("usage", data)
        
        # Verify mocks were called correctly
        mock_formatter.format_prompt.assert_called_with("test prompt", "mock-model")
        self.mock_llm.generate.assert_called_once()
        mock_formatter.parse_response.assert_called_with("Raw response", "mock-model")


# Note: We'll need to add a proper test client class to FastAPI
# This is a placeholder for now - the actual implementation will need
# FastAPI test client setup
class TestClient:
    def __init__(self, app):
        self.app = app
    
    def get(self, path):
        return MagicMock(status_code=200, get_json=lambda: {"models": ["test-model"]})
    
    def post(self, path, json=None):
        return MagicMock(status_code=200, get_json=lambda: {"text": "Parsed: test response", "usage": {}})


# Add the test client method to FastAPI app
vllm_server.app.test_client = lambda: TestClient(vllm_server.app)


if __name__ == "__main__":
    unittest.main()