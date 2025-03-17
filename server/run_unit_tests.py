#!/usr/bin/env python3
"""
Run only the unit tests for the server (no integration tests)
"""

import unittest
from server.test_server import TestPromptFormatter, TestServerUnitTests

if __name__ == "__main__":
    # Create a test suite with just the unit tests
    suite = unittest.TestSuite()
    
    # Add the prompt formatter tests
    suite.addTest(unittest.makeSuite(TestPromptFormatter))
    
    # Add the server unit tests
    suite.addTest(unittest.makeSuite(TestServerUnitTests))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with an error code if tests failed
    import sys
    sys.exit(not result.wasSuccessful())