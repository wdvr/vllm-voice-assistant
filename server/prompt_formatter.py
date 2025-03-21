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
                "system_prompt": "You are a helpful, respectful and honest assistant for a voice interface.",
                "parser": self._parse_llama_response
            },
            
            # Phi-2 model
            "phi": {
                "chat": "System: You are a voice assistant. {system_prompt}\n\nQuestion: {prompt}\n\nAnswer: ",
                "system_prompt": "Keep your responses brief and to the point.",
                "parser": self._parse_general_response
            },
            
            # DeepSeek model
            "deepseek": {
                "chat": "<s><|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n",
                "system_prompt": "You are a voice assistant. Keep answers brief and to the point.",
                "parser": self._parse_general_response
            },
            
            # Default template for any unrecognized model
            "default": {
                "chat": "System: {system_prompt}\n\nUser: {prompt}\n\nAssistant: ",
                "system_prompt": "You are a voice assistant. Provide brief, direct answers.",
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
    
    def format_prompt(self, prompt: str, model_path: str, pre_prompt: Optional[str] = None) -> str:
        """
        Format a prompt for the specified model.
        
        Args:
            prompt: The user's prompt
            model_path: The model path or identifier
            pre_prompt: Optional instructions for how to answer (e.g., "explain like I'm 5")
            
        Returns:
            Formatted prompt string
        """
        model_type = self.detect_model_type(model_path)
        template = self.templates.get(model_type, self.templates["default"])
        
        # Start with the user's original prompt
        final_prompt = prompt
        
        # Always add instruction for concise response, but do this at the end
        # so it doesn't get included in conversation history
        final_prompt = f"{final_prompt}. Answer concisely."
        
        # Get chat template and populate it
        chat_template = template["chat"]
        
        # Check if system prompt should be included
        if "{system_prompt}" in chat_template and "system_prompt" in template:
            # Start with the base system prompt
            system_prompt = template["system_prompt"]
            
            # Incorporate pre-prompt into system message if provided
            if pre_prompt and pre_prompt.strip():
                system_prompt = f"{system_prompt} {pre_prompt}."
            
            # Always add instructions for concise answers to the system prompt
            system_prompt += " Always provide concise, direct answers to questions. If asked for a one-word answer, respond with just one word."
            
            return chat_template.format(
                prompt=final_prompt,
                system_prompt=system_prompt
            )
        
        # Otherwise just use the prompt
        return chat_template.format(prompt=final_prompt)
    
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
        
        # Use the appropriate parser based on model type
        parser = template.get("parser", self._parse_general_response)
        
        # For debugging
        print(f"Raw response: {repr(response)}")
        
        # First pass: model-specific parsing
        parsed = parser(response)
        
        # Second pass: general cleanup for all models
        cleaned = self._clean_response(parsed)
        
        print(f"Parsed response: {repr(cleaned)}")
        
        return cleaned
    
    def _parse_llama_response(self, response: str) -> str:
        """Parse response from Llama models."""
        # Remove any special tokens or formatting specific to Llama
        response = re.sub(r'<\|.*?\|>', '', response).strip()
        return response
    
    def _parse_general_response(self, response: str) -> str:
        """General response parser for models without special requirements."""
        return response.strip()
    
    def _clean_response(self, response: str) -> str:
        """
        Apply general cleanup to any model response to create a clean, concise answer.
        
        Args:
            response: The initially parsed response
            
        Returns:
            A cleaned response suitable for voice output
        """
        if not response or not response.strip():
            return "I don't have an answer for that."
        
        # Remove any remaining special tokens
        cleaned_text = response.strip()
        
        # Check for conversation markers and truncate if found
        conversation_markers = [
            "<|user|>", "<|human|>", "<|assistant|>", "<|bot|>", "<|end|>",
            "\nUser:", "\nHuman:", "\nAssistant:", "\nBot:"
        ]
        
        for marker in conversation_markers:
            pos = cleaned_text.find(marker)
            if pos > 0:  # Only truncate if marker is not at the beginning
                cleaned_text = cleaned_text[:pos].strip()
        
        # Get just the first meaningful line or paragraph
        lines = [line for line in cleaned_text.split('\n') if line.strip()]
        
        if not lines:
            return "I don't have an answer for that."
            
        # For voice output, the first line is usually sufficient
        # But if it's very short, include the second line too
        if len(lines) > 1 and len(lines[0]) < 15:
            first_response = " ".join(lines[:2]).strip()
        else:
            first_response = lines[0].strip()
        
        return first_response


# Singleton instance for easy import
formatter = PromptFormatter()