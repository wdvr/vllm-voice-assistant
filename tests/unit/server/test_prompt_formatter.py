#!/usr/bin/env python3
"""
Unit Tests for Prompt Formatter

These tests verify the prompt formatter functionality that prepares prompts
for different LLM models.
"""

import os
import sys
import unittest

# Get the project root directory and add it to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

# Import the server module
from server.prompt_formatter import formatter


class TestPromptFormatter(unittest.TestCase):
    """Test cases for the PromptFormatter class."""
    
    def test_detect_model_type(self):
        """Test model type detection."""
        # Test Llama model detection
        self.assertEqual(formatter.detect_model_type("llama-2"), "llama")
        self.assertEqual(formatter.detect_model_type("/path/to/meta-llama"), "llama")
        self.assertEqual(formatter.detect_model_type("llama3"), "llama")
        
        # Test Phi model detection
        self.assertEqual(formatter.detect_model_type("phi-2"), "phi")
        self.assertEqual(formatter.detect_model_type("/models/phi"), "phi")
        self.assertEqual(formatter.detect_model_type("microsoft/phi"), "phi")
        
        # Test DeepSeek model detection
        self.assertEqual(formatter.detect_model_type("deepseek"), "deepseek")
        self.assertEqual(formatter.detect_model_type("/models/deepseek-coder"), "deepseek")
        
        # Test default fallback
        self.assertEqual(formatter.detect_model_type("unknown-model"), "default")
    
    def test_format_prompt(self):
        """Test prompt formatting for different model types."""
        # Test Llama format
        llama_prompt = formatter.format_prompt("Hello, world!", "llama")
        self.assertIn("Hello, world!", llama_prompt)
        
        # Test Phi format
        phi_prompt = formatter.format_prompt("Hello, world!", "phi")
        self.assertIn("Hello, world!", phi_prompt)
        self.assertIn("Question:", phi_prompt)
        self.assertIn("Answer:", phi_prompt)
        
        # Test DeepSeek format
        deepseek_prompt = formatter.format_prompt("Hello, world!", "deepseek")
        self.assertIn("Hello, world!", deepseek_prompt)
        self.assertIn("<|user|>", deepseek_prompt)
        self.assertIn("<|assistant|>", deepseek_prompt)
        
        # Test default format
        default_prompt = formatter.format_prompt("Hello, world!", "unknown")
        self.assertEqual(default_prompt, "Hello, world!\n")
    
    def test_parse_response(self):
        """Test response parsing for different model types."""
        # Test Llama response parsing
        llama_response = formatter.parse_response("<|some_token|>Paris is the capital<|end|>", "llama")
        self.assertEqual(llama_response, "Paris is the capital")
        
        # Test general response parsing
        general_response = formatter.parse_response("  Paris is the capital  ", "phi")
        self.assertEqual(general_response, "Paris is the capital")


if __name__ == "__main__":
    unittest.main()