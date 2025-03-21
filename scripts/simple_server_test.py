#!/usr/bin/env python3
"""
A simple test script to interact with the vLLM server

This script provides a straightforward way to send prompts to the vLLM server
and receive responses, useful for quick testing during development.
"""

import requests
import json
import sys
import argparse

def test_query(server_url, prompt, max_tokens=256, temperature=0.7, top_p=0.95):
    """Send a test query to the server."""
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    
    print(f"Sending query to {server_url}:")
    print(f"Prompt: {prompt}")
    
    try:
        response = requests.post(
            f"{server_url}/v1/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\nResponse:")
            print(result["text"])
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Process command line arguments and run the test."""
    parser = argparse.ArgumentParser(description="Test the vLLM server with a prompt")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                       help="Server URL (default: http://localhost:8000)")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?",
                       help="The prompt to send to the server")
    parser.add_argument("--max-tokens", type=int, default=256,
                       help="Maximum number of tokens to generate (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for sampling (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.95,
                       help="Top-p for nucleus sampling (default: 0.95)")
    
    args = parser.parse_args()
    
    success = test_query(
        args.server,
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_p
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()