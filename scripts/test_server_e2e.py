#!/usr/bin/env python3
"""
End-to-end test script for the vLLM server

This script:
1. Starts the server with a test model
2. Sends a test query
3. Verifies the response
"""

import os
import sys
import subprocess
import time
import requests
import json
import logging
import argparse
import signal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global variables
SERVER_PROCESS = None
SERVER_URL = "http://localhost:8000"
TEST_MODEL_PATH = "./models/phi-2"
DEFAULT_GPU_UTIL = 0.8


def start_server(model_path, gpu_util=DEFAULT_GPU_UTIL, port=8000):
    """Start the vLLM server in a separate process."""
    global SERVER_PROCESS
    
    try:
        # Get the project root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        server_script = os.path.join(root_dir, "server", "vllm_server.py")
        
        # Get the virtual environment's Python
        venv_python = os.path.join(root_dir, "venv", "bin", "python")
        python_exec = venv_python if os.path.exists(venv_python) else sys.executable
        
        # Command to run
        cmd = [
            python_exec,
            server_script,
            "--model", model_path,
            "--gpu-memory-utilization", str(gpu_util),
            "--port", str(port)
        ]
        
        # Start the server process
        logger.info(f"Starting server with command: {' '.join(cmd)}")
        SERVER_PROCESS = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            cwd=root_dir
        )
        
        # Wait for server to start (simple polling)
        logger.info("Waiting for server to start...")
        for i in range(60):  # Wait up to 60 seconds
            try:
                response = requests.get(f"{SERVER_URL}/v1/models", timeout=1)
                if response.status_code == 200:
                    logger.info("Server started successfully")
                    return True
            except requests.RequestException:
                # Print progress every 10 seconds
                if i % 10 == 0 and i > 0:
                    logger.info(f"Still waiting for server to start... ({i}s)")
                
                # Check if server process has terminated
                if SERVER_PROCESS.poll() is not None:
                    stdout, stderr = SERVER_PROCESS.communicate()
                    logger.error("Server process terminated unexpectedly")
                    logger.error(f"STDOUT: {stdout}")
                    logger.error(f"STDERR: {stderr}")
                    return False
                
                time.sleep(1)
        
        logger.error("Failed to start server within timeout")
        return False
    
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        stop_server()
        return False


def stop_server():
    """Stop the vLLM server process."""
    global SERVER_PROCESS
    if SERVER_PROCESS:
        logger.info("Stopping server...")
        SERVER_PROCESS.terminate()
        try:
            SERVER_PROCESS.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not terminate gracefully, killing...")
            SERVER_PROCESS.kill()
        
        SERVER_PROCESS = None
        logger.info("Server stopped")


def test_query(prompt="What is the capital of France?"):
    """Send a test query to the server and verify response."""
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95
    }
    
    logger.info(f"Sending test query: {prompt}")
    
    try:
        response = requests.post(
            f"{SERVER_URL}/v1/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            logger.info(f"Got response: {result['text']}")
            
            # Test that we got a valid response
            assert "text" in result, "Response doesn't contain 'text' field"
            assert isinstance(result["text"], str), "Response text is not a string"
            assert len(result["text"]) > 0, "Response text is empty"
            
            logger.info("Test query successful")
            return True
        else:
            logger.error(f"Request failed with status {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Error testing query: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test the vLLM server end-to-end")
    parser.add_argument("--model", type=str, default=TEST_MODEL_PATH,
                        help=f"Path to the model (default: {TEST_MODEL_PATH})")
    parser.add_argument("--gpu-util", type=float, default=DEFAULT_GPU_UTIL,
                        help=f"GPU memory utilization (default: {DEFAULT_GPU_UTIL})")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on (default: 8000)")
    
    args = parser.parse_args()
    
    # Set up signal handlers to clean up server process on exit
    def signal_handler(sig, frame):
        logger.info("Test interrupted, stopping server...")
        stop_server()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    success = False
    try:
        # Start the server
        if not start_server(args.model, args.gpu_util, args.port):
            logger.error("Failed to start server, exiting.")
            sys.exit(1)
        
        # Wait a bit to let the server initialize fully
        time.sleep(5)
        
        # Test a query
        success = test_query()
        
        if success:
            logger.info("End-to-end test passed!")
        else:
            logger.error("End-to-end test failed!")
    
    finally:
        # Clean up
        stop_server()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()