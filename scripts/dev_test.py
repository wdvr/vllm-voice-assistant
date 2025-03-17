#!/usr/bin/env python3
"""
Development Test Script

This script allows running both server and client on the same machine for development testing.
It simulates the voice assistant interaction without requiring separate hardware.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Optional

import requests


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Server process
server_process = None
SERVER_URL = "http://localhost:8000"


def start_server(model_path: str, gpu_util: float = 0.9):
    """Start the vLLM server in a separate process."""
    global server_process
    
    try:
        # Construct the server command
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         "server", "vllm_server.py"),
            "--model", model_path,
            "--gpu-memory-utilization", str(gpu_util)
        ]
        
        # Start the server process
        logger.info(f"Starting server with command: {' '.join(cmd)}")
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Wait for server to start (simple polling)
        logger.info("Waiting for server to start...")
        for _ in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(f"{SERVER_URL}/v1/models", timeout=1)
                if response.status_code == 200:
                    logger.info("Server started successfully")
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
            
        logger.error("Failed to start server within timeout")
        return False
            
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return False


def stop_server():
    """Stop the vLLM server process."""
    global server_process
    if server_process:
        logger.info("Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not terminate gracefully, killing...")
            server_process.kill()
        server_process = None
        logger.info("Server stopped")


def send_query(prompt: str):
    """Send a text query to the LLM server."""
    try:
        # Prepare the request
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95
        }
        
        # Send the request
        response = requests.post(
            f"{SERVER_URL}/v1/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=30
        )
        
        # Process the response
        if response.status_code == 200:
            return response.json()["text"]
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return f"Error: Failed to get response from server. Status code: {response.status_code}"
    except Exception as e:
        logger.error(f"Request error: {e}")
        return f"Error: {str(e)}"


def interactive_mode():
    """Run in interactive CLI mode for testing."""
    logger.info("Starting interactive mode. Type 'exit' to quit.")
    
    while True:
        # Get user input
        prompt = input("\nEnter your query (or 'exit' to quit): ")
        
        if prompt.lower() in ["exit", "quit"]:
            break
            
        if not prompt.strip():
            continue
        
        logger.info("Sending query to server...")
        response = send_query(prompt)
        
        print("\n--- LLM Response ---")
        print(response)
        print("-------------------")
    
    logger.info("Exiting interactive mode")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Voice Assistant Development Test Script")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to the model or model name")
    parser.add_argument("--gpu-util", type=float, default=0.9,
                        help="GPU memory utilization (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    try:
        # Start the server
        if not start_server(args.model, args.gpu_util):
            logger.error("Failed to start server. Exiting.")
            sys.exit(1)
        
        # Run interactive mode
        interactive_mode()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting...")
    finally:
        # Clean up
        stop_server()
        logger.info("Test script exited")


if __name__ == "__main__":
    main()