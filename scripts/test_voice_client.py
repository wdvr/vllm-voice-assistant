#!/usr/bin/env python3
"""
Voice Client Test Script

This script helps test the voice assistant client on macOS using the mock LLM server.
It can run in two modes:
1. Automated test mode: Sends predetermined test prompts to test basic functionality
2. Interactive mode: Allows user to interact with the voice client using the microphone
"""

import argparse
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
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Server process
server_process = None
client_process = None
SERVER_URL = "http://localhost:8000"


def start_mock_server(host: str = "localhost", port: int = 8000):
    """Start the mock LLM server in a separate process."""
    global server_process
    
    try:
        # Get the project root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        server_script = os.path.join(root_dir, "scripts", "mock_llm_server.py")
        
        # Use the virtual environment's Python if available
        venv_python = os.path.join(root_dir, "venv", "bin", "python")
        python_exec = venv_python if os.path.exists(venv_python) else sys.executable
        
        # Construct the server command
        cmd = [
            python_exec,
            server_script,
            "--host", host,
            "--port", str(port),
            "--model", "mock-model"
        ]
        
        # Start the server process
        logger.info(f"Starting mock server with command: {' '.join(cmd)}")
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            cwd=root_dir  # Set working directory to project root
        )
        
        # Wait for server to start (simple polling)
        logger.info("Waiting for server to start...")
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(f"{SERVER_URL}/v1/models", timeout=1)
                if response.status_code == 200:
                    logger.info("Mock server started successfully")
                    return True
            except requests.RequestException:
                # Check if the server process has terminated unexpectedly
                if server_process.poll() is not None:
                    stdout, stderr = server_process.communicate()
                    logger.error("Server process terminated unexpectedly!")
                    logger.error(f"STDOUT: {stdout}")
                    logger.error(f"STDERR: {stderr}")
                    return False
            time.sleep(1)
            
        logger.error("Failed to start mock server within timeout")
        return False
            
    except Exception as e:
        logger.error(f"Failed to start mock server: {e}")
        return False


def start_voice_client(server_url: str = SERVER_URL, device: str = "cpu"):
    """Start the voice client in a separate process."""
    global client_process
    
    try:
        # Get the project root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        client_script = os.path.join(root_dir, "client", "voice_client.py")
        
        # Use the virtual environment's Python if available
        venv_python = os.path.join(root_dir, "venv", "bin", "python")
        python_exec = venv_python if os.path.exists(venv_python) else sys.executable
        
        # Construct the client command
        cmd = [
            python_exec,
            client_script,
            "--server", server_url,
            "--device", device
        ]
        
        # Start the client process
        logger.info(f"Starting voice client with command: {' '.join(cmd)}")
        client_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            cwd=root_dir  # Set working directory to project root
        )
        
        # Give it a moment to initialize
        time.sleep(2)
        
        # Check if client is still running
        if client_process.poll() is not None:
            stdout, stderr = client_process.communicate()
            logger.error("Client process terminated unexpectedly!")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            return False
            
        logger.info("Voice client started successfully")
        return True
            
    except Exception as e:
        logger.error(f"Failed to start voice client: {e}")
        return False


def stop_server():
    """Stop the mock server process."""
    global server_process
    if server_process:
        logger.info("Stopping mock server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not terminate gracefully, killing...")
            server_process.kill()
        server_process = None
        logger.info("Mock server stopped")


def stop_client():
    """Stop the voice client process."""
    global client_process
    if client_process:
        logger.info("Stopping voice client...")
        client_process.terminate()
        try:
            client_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Client did not terminate gracefully, killing...")
            client_process.kill()
        client_process = None
        logger.info("Voice client stopped")


def monitor_client_output():
    """Thread function to monitor and print client output."""
    if client_process:
        for line in client_process.stdout:
            print(f"CLIENT: {line.strip()}")


def simulate_enter_keypress():
    """Simulate pressing Enter in the client process."""
    if client_process and client_process.stdin:
        logger.info("Simulating Enter keypress...")
        client_process.stdin.write("\n")
        client_process.stdin.flush()


def interactive_test():
    """Run the client in interactive mode."""
    logger.info("Starting interactive test mode")
    logger.info("The voice client is now running. Follow the prompts in the client window.")
    
    # Start a thread to monitor client output
    monitor_thread = threading.Thread(target=monitor_client_output, daemon=True)
    monitor_thread.start()
    
    try:
        while True:
            cmd = input("\nCommands: [r] record, [q] quit: ")
            
            if cmd.lower() == 'q':
                break
            elif cmd.lower() == 'r':
                # Simulate pressing Enter to start recording
                simulate_enter_keypress()
                print("Recording started. Speak into your microphone...")
                print("Press Enter when done speaking...")
                input()
                # Simulate pressing Enter to stop recording
                simulate_enter_keypress()
            else:
                print("Unknown command")
                
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        logger.info("Ending interactive test")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Voice Client Test Script")
    parser.add_argument("--mode", choices=["interactive"], default="interactive",
                        help="Test mode: interactive (uses microphone)")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Host for the mock server")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the mock server")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for Whisper model (cpu, mps for Mac)")
    
    args = parser.parse_args()
    
    # Get project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        # Try to activate virtual environment
        venv_python = os.path.join(root_dir, "venv", "bin", "python")
        if os.path.exists(venv_python):
            logger.info(f"Restarting script with virtual environment Python: {venv_python}")
            os.execl(venv_python, venv_python, *sys.argv)
    
    try:
        # Start the mock server
        server_url = f"http://{args.host}:{args.port}"
        if not start_mock_server(args.host, args.port):
            logger.error("Failed to start mock server. Exiting.")
            sys.exit(1)
        
        # Start the voice client
        if not start_voice_client(server_url, args.device):
            logger.error("Failed to start voice client. Exiting.")
            stop_server()
            sys.exit(1)
        
        # Run the selected test mode
        if args.mode == "interactive":
            interactive_test()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting...")
    finally:
        # Clean up
        stop_client()
        stop_server()
        logger.info("Test script exited")


if __name__ == "__main__":
    main()