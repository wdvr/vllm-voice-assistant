#!/usr/bin/env python3
"""
Prompt Formatter for Different LLM Models

This module provides prompt formatting functionality for different language models.
Each model has specific prompt templates and requirements for optimal performance.
"""

import os
import re
from typing import Dict, Any, Optional


class PromptFormatter:
    """
    Formats prompts appropriately for different language models.
    """
    
    def __init__(self):
        # Model-specific templates
        self.templates = {
            # Llama family of models
            "llama": {
                "chat": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n",
                "system_prompt": "You are a helpful, respectful and honest assistant.",
                "parser": self._parse_llama_response
            },
            
            # Phi-2 model
            "phi": {
                "chat": "Instruct: {prompt}\nOutput: ",
                "parser": self._parse_general_response
            },
            
            # DeepSeek model
            "deepseek": {
                "chat": "<s><|user|>\n{prompt}\n<|assistant|>\n",
                "system_prompt": "You are DeepSeek, an AI assistant by DeepSeek Company. Always be helpful, harmless, and honest.",
                "parser": self._parse_general_response
            },
            
            # Default template for any unrecognized model
            "default": {
                "chat": "{prompt}\n",
                "parser": self._parse_general_response
            }
        }
        
        # Aliases for model matching
        self.model_aliases = {
            "llama": ["llama", "meta-llama", "meta-llama-3", "llama-2", "llama3"],
            "phi": ["phi", "phi-2", "phi-3", "microsoft/phi"],
            "deepseek": ["deepseek", "deepseek-coder", "deepseek-llm"],
        }
    
    def detect_model_type(self, model_path: str) -> str:
        """
        Detect the model type from the model path.
        
        Args:
            model_path: Path to the model or model identifier
            
        Returns:
            String identifying the model type
        """
        model_path_lower = model_path.lower()
        
        # Check if model_path matches any aliases
        for model_type, aliases in self.model_aliases.items():
            if any(alias in model_path_lower for alias in aliases):
                return model_type
                
        # If no match found, return default
        return "default"
    
    def format_prompt(self, prompt: str, model_path: str) -> str:
        """
        Format a prompt for the specified model.
        
        Args:
            prompt: The user's prompt
            model_path: The model path or identifier
            
        Returns:
            Formatted prompt string
        """
        model_type = self.detect_model_type(model_path)
        template = self.templates.get(model_type, self.templates["default"])
        
        # Get chat template and populate it
        chat_template = template["chat"]
        
        # Check if system prompt should be included
        if "{system_prompt}" in chat_template and "system_prompt" in template:
            return chat_template.format(
                prompt=prompt,
                system_prompt=template["system_prompt"]
            )
        
        # Otherwise just use the prompt
        return chat_template.format(prompt=prompt)
    
    def parse_response(self, response: str, model_path: str) -> str:
        """
        Parse the response from the model, removing any special tokens or formatting.
        
        Args:
            response: Raw response from the model
            model_path: The model path or identifier
            
        Returns:
            Cleaned response text
        """
        model_type = self.detect_model_type(model_path)
        template = self.templates.get(model_type, self.templates["default"])
        
        # Use the appropriate parser
        parser = template.get("parser", self._parse_general_response)
        return parser(response)
    
    def _parse_llama_response(self, response: str) -> str:
        """Parse response from Llama models."""
        # Remove any special tokens or formatting specific to Llama
        response = re.sub(r'<\|.*?\|>', '', response).strip()
        return response
    
    def _parse_general_response(self, response: str) -> str:
        """General response parser for models without special requirements."""
        return response.strip()


# Singleton instance for easy import
formatter = PromptFormatter()