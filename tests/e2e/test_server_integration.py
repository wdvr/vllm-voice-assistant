#!/usr/bin/env python3
"""
Integration Tests for vLLM Server

These tests verify the server's functionality including:
1. Loading a model
2. Generating completions
3. End-to-end operation
"""

import os
import sys
import unittest
import json
import requests
import subprocess
import time
import logging

# Get the project root directory and add it to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# Import the server modules
from server.prompt_formatter import formatter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test settings
TEST_MODEL_PATH = "./models/phi-2"  # Path to a small test model
SERVER_URL = "http://localhost:8765"  # Use a different port for testing


class TestServerIntegration(unittest.TestCase):
    """Integration tests for the vLLM server"""
    
    server_process = None
    
    @classmethod
    def setUpClass(cls):
        """Start the server for integration tests"""
        if not os.path.exists(TEST_MODEL_PATH):
            logger.warning(f"Test model not found at {TEST_MODEL_PATH}, skipping integration tests")
            return
        
        try:
            # Start the server in a separate process
            cls.server_process = subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(root_dir, "server", "vllm_server.py"),
                    "--model", TEST_MODEL_PATH,
                    "--port", "8765",
                    "--gpu-memory-utilization", "0.9"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for the server to start
            logger.info("Waiting for server to start...")
            for _ in range(60):  # Wait up to 60 seconds
                try:
                    response = requests.get(f"{SERVER_URL}/v1/models", timeout=1)
                    if response.status_code == 200:
                        logger.info("Server started successfully")
                        return
                except requests.RequestException:
                    pass
                time.sleep(1)
                
            logger.error("Failed to start server for integration tests")
            cls.tearDownClass()
            
        except Exception as e:
            logger.error(f"Error setting up server: {e}")
            cls.tearDownClass()
    
    @classmethod
    def tearDownClass(cls):
        """Stop the server after integration tests"""
        if cls.server_process:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()
    
    @unittest.skipIf(not os.path.exists(TEST_MODEL_PATH), f"Test model not found at {TEST_MODEL_PATH}")
    def test_end_to_end(self):
        """Test an end-to-end request to the server"""
        if not self.server_process:
            self.skipTest("Server not running")
        
        # Check if SERVER_PROCESS failed to start properly
        if self.server_process.poll() is not None:
            stdout, stderr = self.server_process.communicate()
            logger.error("Server process terminated unexpectedly")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            self.fail("Server process terminated unexpectedly")
        
        # Wait a bit for server to fully initialize
        time.sleep(5)
        
        # Test the model listing endpoint
        try:
            response = requests.get(f"{SERVER_URL}/v1/models", timeout=2)
            self.assertEqual(response.status_code, 200)
            models = response.json()
            self.assertIn("models", models)
            
            # Send a test prompt
            prompt = "What is the capital of France?"
            
            headers = {"Content-Type": "application/json"}
            data = {
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.95
            }
            
            logger.info(f"Sending test prompt: {prompt}")
            response = requests.post(
                f"{SERVER_URL}/v1/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=60  # Longer timeout for actual model inference
            )
            
            self.assertEqual(response.status_code, 200)
            result = response.json()
            
            # Check that we got a text response
            self.assertIn("text", result)
            self.assertIsInstance(result["text"], str)
            self.assertTrue(len(result["text"]) > 0)
            
            # Check that we got usage information
            self.assertIn("usage", result)
            
            logger.info(f"Got response: {result['text']}")
            
        except requests.RequestException as e:
            # Check server status more carefully
            if self.server_process.poll() is not None:
                stdout, stderr = self.server_process.communicate()
                logger.error(f"Server failed during test. STDOUT: {stdout[:500]}")
                logger.error(f"STDERR: {stderr[:500]}")
                self.fail(f"Server crashed: {e}")
            else:
                # Server is running but request failed
                self.fail(f"Request failed: {e}")


if __name__ == "__main__":
    unittest.main()