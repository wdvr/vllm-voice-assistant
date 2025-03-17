#!/usr/bin/env python3
"""
Main Test Runner for vLLM Voice Assistant

This script runs all tests for the voice assistant project:
- Unit tests
- End-to-end tests
"""

import argparse
import os
import subprocess
import sys
import unittest
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_unit_tests():
    """Run all unit tests."""
    logger.info("Running unit tests...")
    
    # Get the current directory and project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    unit_test_dir = os.path.join(current_dir, 'unit')
    
    # Import the test modules directly for now
    client_tests_success = run_client_unit_tests()
    
    return client_tests_success


def run_client_unit_tests():
    """Run client unit tests specifically."""
    logger.info("Running client unit tests...")
    
    # Use subprocess to run the test directly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    test_script = os.path.join(current_dir, 'unit', 'client', 'test_voice_client.py')
    
    # Use the virtual environment's Python if available
    venv_python = os.path.join(project_root, "venv", "bin", "python")
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable
    
    cmd = [python_exec, test_script]
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        logger.error("Client unit tests failed")
        return False


def run_e2e_tests(interactive=False):
    """Run end-to-end tests."""
    logger.info("Running end-to-end tests...")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Use the virtual environment's Python if available
    venv_python = os.path.join(project_root, "venv", "bin", "python")
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable
    
    # Build the command
    e2e_script = os.path.join(current_dir, 'e2e', 'test_voice_assistant.py')
    mode = "both" if interactive else "auto"
    
    cmd = [
        python_exec,
        e2e_script,
        "--mode", mode
    ]
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        logger.error("End-to-end tests failed")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run voice assistant tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--interactive", action="store_true", 
                      help="Run e2e tests in interactive mode")
    
    args = parser.parse_args()
    
    # If no specific tests are selected, run all
    run_all = args.all or (not args.unit and not args.e2e)
    success = True
    
    if run_all or args.unit:
        unit_success = run_unit_tests()
        if not unit_success:
            success = False
            logger.error("Unit tests failed")
        else:
            logger.info("Unit tests passed")
    
    if run_all or args.e2e:
        e2e_success = run_e2e_tests(args.interactive)
        if not e2e_success:
            success = False
            logger.error("End-to-end tests failed")
        else:
            logger.info("End-to-end tests passed")
    
    if success:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())