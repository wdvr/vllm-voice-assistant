#!/usr/bin/env python3
"""
Run only the unit tests for the client

This is a legacy script that now redirects to the new test structure.
Please use `python tests/run_tests.py --unit` instead.
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    print("WARNING: This script is deprecated.")
    print("Please use 'python tests/run_tests.py --unit' instead.")
    print("Redirecting to new test runner...")
    
    # Get the project root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(root_dir)
    
    # Path to new test runner
    test_runner = os.path.join(project_root, "tests", "run_tests.py")
    
    # Use the virtual environment's Python if available
    venv_python = os.path.join(project_root, "venv", "bin", "python")
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable
    
    # Run the new test runner
    try:
        result = subprocess.run([python_exec, test_runner, "--unit"], check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)