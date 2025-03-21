#!/usr/bin/env python3
"""
Unit Tests for Prompt Formatter with GGUF Models

These tests verify the prompt formatter functionality with a small GGUF-based LLM model.
"""

import os
import sys
import unittest

# Get the project root directory and add it to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)

# Import the server module
from server.prompt_formatter import formatter

# Simple test LLM using ctransformers
try:
    import ctransformers
    from ctransformers import AutoModelForCausalLM
    HAVE_CTRANSFORMERS = True
except ImportError:
    HAVE_CTRANSFORMERS = False


@unittest.skipIf(not HAVE_CTRANSFORMERS, "ctransformers package not installed")
class TestPromptFormatterWithGGUF(unittest.TestCase):
    """Test cases for the PromptFormatter class with a real GGUF model."""
    
    def setUp(self):
        """Set up a small model for testing."""
        # Try to load a small test model
        try:
            model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
            model_file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
            
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                model_file=model_file,
                model_type="llama",
                gpu_layers=0  # Use CPU only for testing
            )
            self.model_loaded = True
        except Exception as e:
            try:
                # Fallback to a smaller model
                self.llm = AutoModelForCausalLM.from_pretrained(
                    "marella/gpt-2-ggml", 
                    model_file="ggml-model-q4_0.bin",
                    model_type="gpt2"
                )
                self.model_loaded = True
            except Exception:
                self.model_loaded = False
    
    def test_model_response_with_llama_template(self):
        """Test using the Llama template with a real model."""
        if not self.model_loaded:
            self.skipTest("Could not load any test model")
        
        prompt = "What is the capital of France?"
        model_type = "llama"
        
        # Format using our prompt formatter
        formatted_prompt = formatter.format_prompt(prompt, model_type)
        
        # Generate with the model
        raw_response = self.llm(formatted_prompt, max_new_tokens=50)
        
        # Parse the response using our formatter
        parsed = formatter.parse_response(raw_response, model_type)
        
        # Verify the response is non-empty (actual content is model-dependent)
        self.assertTrue(len(parsed) > 0)
    
    def test_model_response_with_phi_template(self):
        """Test using the Phi template with a real model."""
        if not self.model_loaded:
            self.skipTest("Could not load any test model")
        
        prompt = "What is the capital of France?"
        model_type = "phi"
        
        # Format using our prompt formatter
        formatted_prompt = formatter.format_prompt(prompt, model_type)
        
        # Generate with the model
        raw_response = self.llm(formatted_prompt, max_new_tokens=50)
        
        # Parse the response using our formatter
        parsed = formatter.parse_response(raw_response, model_type)
        
        # Verify the response is non-empty (actual content is model-dependent)
        self.assertTrue(len(parsed) > 0)
    
    def test_model_response_with_deepseek_template(self):
        """Test using the DeepSeek template with a real model."""
        if not self.model_loaded:
            self.skipTest("Could not load any test model")
        
        prompt = "What is the capital of France?"
        model_type = "deepseek"
        
        # Format using our prompt formatter
        formatted_prompt = formatter.format_prompt(prompt, model_type)
        
        # Generate with the model
        raw_response = self.llm(formatted_prompt, max_new_tokens=50)
        
        # Parse the response using our formatter
        parsed = formatter.parse_response(raw_response, model_type)
        
        # Verify the response is non-empty (actual content is model-dependent)
        self.assertTrue(len(parsed) > 0)


if __name__ == "__main__":
    unittest.main()