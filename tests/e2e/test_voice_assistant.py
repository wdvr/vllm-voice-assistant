#!/usr/bin/env python3
"""
End-to-End Test for Voice Assistant

This script runs a complete end-to-end test of the voice assistant system:
1. Starts the mock LLM server
2. Runs automated tests for the voice client
3. Verifies all components are working together
"""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
import queue
import json
from typing import Optional, List, Dict, Any

import requests
import sounddevice as sd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global variables
server_process = None
client_module = None
SERVER_URL = "http://localhost:8123"
test_queue = queue.Queue()
TEST_PROMPTS = [
    "Hello",
    "What's the weather like today?",
    "Tell me a joke",
    "What's your name?",
    "Goodbye"
]
test_results = []


def start_mock_server(host: str = "localhost", port: int = 8123):
    """Start the mock LLM server in a separate process."""
    global server_process
    
    try:
        # Get the project root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        
        # Wait for server to start
        logger.info("Waiting for mock server to start...")
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


def import_client_module():
    """Import the client module to use its functions directly."""
    global client_module
    
    # Get the project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Add it to sys.path if not already there
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    # Import the client module
    try:
        import client.voice_client as client_module
        logger.info("Successfully imported client module")
    except ImportError as e:
        logger.error(f"Failed to import client module: {e}")
        sys.exit(1)


def send_request_to_server(prompt: str) -> dict:
    """Send a request directly to the mock server."""
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95
        }
        
        response = requests.post(
            f"{SERVER_URL}/v1/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return {"error": f"Status code: {response.status_code}"}
    except Exception as e:
        logger.error(f"Request error: {e}")
        return {"error": str(e)}


def run_server_tests():
    """Run tests directly against the server."""
    logger.info("Running direct server tests...")
    results = []
    
    # Test model listing
    try:
        response = requests.get(f"{SERVER_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            results.append(("Model List", "PASS", response.json()))
        else:
            results.append(("Model List", "FAIL", response.status_code))
    except Exception as e:
        results.append(("Model List", "FAIL", str(e)))
    
    # Test model info
    try:
        response = requests.get(f"{SERVER_URL}/v1/model/info", timeout=5)
        if response.status_code == 200:
            results.append(("Model Info", "PASS", response.json()))
        else:
            results.append(("Model Info", "FAIL", response.status_code))
    except Exception as e:
        results.append(("Model Info", "FAIL", str(e)))
    
    # Test text completion with various prompts
    for prompt in TEST_PROMPTS:
        try:
            result = send_request_to_server(prompt)
            if "text" in result:
                results.append((f"Text Completion: '{prompt}'", "PASS", result["text"]))
            else:
                results.append((f"Text Completion: '{prompt}'", "FAIL", result))
        except Exception as e:
            results.append((f"Text Completion: '{prompt}'", "FAIL", str(e)))
    
    # Display results
    logger.info("Server Test Results:")
    for test, status, details in results:
        logger.info(f"  {test}: {status}")
        if status == "PASS":
            logger.debug(f"    Details: {details}")
        else:
            logger.error(f"    Details: {details}")
    
    return all(status == "PASS" for _, status, _ in results)


def initialize_client():
    """Initialize the client components."""
    logger.info("Initializing client components...")
    
    # Set up the server URL
    client_module.SERVER_URL = SERVER_URL
    
    # Initialize text-to-speech engine
    logger.info("Initializing text-to-speech engine...")
    client_module.initialize_tts()
    
    # We'll use a simpler approach without loading Whisper for automated testing
    logger.info("Setting up mock transcriber...")
    client_module.transcriber = lambda audio_data: {"text": test_queue.get()}


def audio_callback(indata, frames, time, status):
    """Callback for audio recording (not used in automated tests)."""
    pass


def test_client_tts():
    """Test the text-to-speech functionality."""
    logger.info("Testing text-to-speech...")
    try:
        client_module.text_to_speech("This is a test of the text to speech system.")
        return True
    except Exception as e:
        logger.error(f"TTS test failed: {e}")
        return False


def test_client_llm_communication():
    """Test the client's ability to communicate with the LLM server."""
    logger.info("Testing LLM communication...")
    for prompt in TEST_PROMPTS:
        try:
            response = client_module.send_to_llm(prompt)
            logger.info(f"Prompt: '{prompt}', Response: '{response}'")
            if not response or "Error" in response:
                logger.error(f"LLM communication failed for prompt: '{prompt}'")
                return False
        except Exception as e:
            logger.error(f"LLM communication test failed: {e}")
            return False
    return True


def test_client_full_pipeline():
    """Test the full client pipeline with simulated audio input."""
    logger.info("Testing full client pipeline...")
    results = []
    
    for prompt in TEST_PROMPTS:
        try:
            # Simulate audio data by putting the prompt in the test queue
            test_queue.put(prompt)
            
            # In a real setup, we'd do:
            # 1. Record audio
            # 2. Transcribe it
            # 3. Send to LLM
            # 4. Get response
            # 5. Convert to speech
            
            # Here, we'll just simulate the key steps:
            logger.info(f"Simulated transcription: '{prompt}'")
            
            # Send to LLM
            llm_response = client_module.send_to_llm(prompt)
            logger.info(f"LLM Response: '{llm_response}'")
            
            # Text to speech (just log, don't actually speak in automated test)
            logger.info(f"Converting to speech: '{llm_response}'")
            
            results.append((prompt, "PASS", llm_response))
        except Exception as e:
            logger.error(f"Full pipeline test failed for '{prompt}': {e}")
            results.append((prompt, "FAIL", str(e)))
    
    # Display results
    logger.info("Full Pipeline Test Results:")
    for prompt, status, details in results:
        logger.info(f"  Prompt '{prompt}': {status}")
        if status == "PASS":
            logger.info(f"    Response: {details}")
        else:
            logger.error(f"    Error: {details}")
    
    return all(status == "PASS" for _, status, _ in results)


def run_automated_tests():
    """Run a series of automated tests."""
    logger.info("Running automated tests...")
    
    # Test direct server communication
    server_test_success = run_server_tests()
    logger.info(f"Server tests {'PASSED' if server_test_success else 'FAILED'}")
    
    # Initialize client components
    initialize_client()
    
    # Test TTS
    tts_test_success = test_client_tts()
    logger.info(f"TTS tests {'PASSED' if tts_test_success else 'FAILED'}")
    
    # Test LLM communication
    llm_test_success = test_client_llm_communication()
    logger.info(f"LLM communication tests {'PASSED' if llm_test_success else 'FAILED'}")
    
    # Test full pipeline
    pipeline_success = test_client_full_pipeline()
    logger.info(f"Full pipeline tests {'PASSED' if pipeline_success else 'FAILED'}")
    
    # Overall result
    all_passed = server_test_success and tts_test_success and llm_test_success and pipeline_success
    
    return all_passed


def interactive_test():
    """Run an interactive test where the user can speak to the assistant."""
    logger.info("Starting interactive test session...")
    
    # Get client ready
    initialize_client()
    
    # Initialize the real Whisper model
    device = "mps" if is_mps_available() else "cpu"
    logger.info(f"Initializing Whisper with device: {device}")
    client_module.initialize_whisper(device=device)
    
    print("\n============================================")
    print("   Voice Assistant Interactive Test")
    print("============================================")
    print("This test uses your computer's microphone to interact")
    print("with the voice assistant. The mock server will respond")
    print("with pre-defined responses based on keywords in your query.")
    print("\nCommands:")
    print("  r - Start recording (speak, then press Enter to stop)")
    print("  q - Quit the test")
    print("============================================\n")
    
    while True:
        cmd = input("Enter command (r = record, q = quit): ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 'r':
            # Start recording
            print("Recording... (speak now, press Enter when done)")
            client_module.is_recording = True
            stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                callback=client_module.audio_callback
            )
            stream.start()
            
            # Wait for user to press Enter
            input()
            
            # Stop recording
            stream.stop()
            stream.close()
            client_module.is_recording = False
            
            # Get audio data from queue
            audio_data = []
            while not client_module.audio_queue.empty():
                audio_data.append(client_module.audio_queue.get())
            
            if not audio_data:
                print("No audio recorded!")
                continue
            
            # Process the audio
            print("Transcribing audio...")
            transcription = client_module.transcribe_audio(audio_data)
            if not transcription:
                print("Failed to transcribe audio")
                continue
            
            print(f"Transcription: {transcription}")
            
            # Send to LLM
            print("Sending to LLM...")
            response = client_module.send_to_llm(transcription)
            print(f"LLM Response: {response}")
            
            # Text to speech
            print("Converting to speech...")
            client_module.text_to_speech(response)
        else:
            print("Unknown command")
    
    print("Interactive test completed")


def is_mps_available():
    """Check if Metal Performance Shaders are available."""
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True
    except (ImportError, AttributeError):
        pass
    return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Voice Assistant E2E Test")
    parser.add_argument("--mode", choices=["auto", "interactive", "both"],
                      default="both", help="Test mode")
    parser.add_argument("--host", type=str, default="localhost",
                      help="Host for the mock server")
    parser.add_argument("--port", type=int, default=8123,
                      help="Port for the mock server")
    
    args = parser.parse_args()
    
    # Get project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        # Try to activate virtual environment
        venv_python = os.path.join(root_dir, "venv", "bin", "python")
        if os.path.exists(venv_python):
            logger.info(f"Restarting script with virtual environment Python: {venv_python}")
            os.execl(venv_python, venv_python, *sys.argv)
    
    logger.info(f"Starting E2E test in {args.mode} mode")
    
    try:
        # Start the mock server
        if not start_mock_server(args.host, args.port):
            logger.error("Failed to start mock server. Exiting.")
            sys.exit(1)
        
        # Import the client module
        import_client_module()
        
        # Run tests based on the mode
        if args.mode in ["auto", "both"]:
            all_tests_passed = run_automated_tests()
            if args.mode == "auto":
                sys.exit(0 if all_tests_passed else 1)
        
        if args.mode in ["interactive", "both"]:
            interactive_test()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up
        stop_server()
        logger.info("E2E test completed")


if __name__ == "__main__":
    main()