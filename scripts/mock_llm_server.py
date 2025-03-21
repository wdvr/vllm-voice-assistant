#!/usr/bin/env python3
"""
Mock vLLM Server for Voice Assistant Development

This script provides a lightweight mock implementation of the vLLM server
for development and testing on macOS or systems without GPU.
"""

import argparse
import logging
import os
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global variables
model_path = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Replaces deprecated @app.on_event("startup") and @app.on_event("shutdown").
    """
    # Startup
    logger.info("Starting up mock vLLM server...")
    yield
    # Shutdown
    logger.info("Shutting down mock vLLM server...")
    
    # Clean up any resources if needed
    try:
        # Since this is a mock server, we don't actually have NCCL process groups,
        # but we'll include similar code structure for consistency
        logger.debug("Cleaning up mock server resources")
    except Exception as e:
        logger.warning(f"Error during mock server cleanup: {e}")


# Initialize FastAPI
app = FastAPI(title="Mock Voice Assistant vLLM API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerationRequest(BaseModel):
    """Request body for text generation."""
    prompt: str = Field(..., description="The prompt to generate text from")
    pre_prompt: Optional[str] = Field(None, description="Optional instructions for how to answer (e.g., 'explain like I'm 5')")
    max_tokens: int = Field(512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.95, description="Top-p sampling parameter")
    presence_penalty: float = Field(0.0, description="Presence penalty")
    frequency_penalty: float = Field(0.0, description="Frequency penalty")


class GenerationResponse(BaseModel):
    """Response from text generation."""
    text: str = Field(..., description="Generated text")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Name of the loaded model")
    max_model_len: int = Field(..., description="Maximum context length")
    gpu_memory_usage: str = Field(..., description="GPU memory usage")


# Startup event handler has been replaced with the lifespan context manager


@app.get("/v1/models", response_model=Dict[str, List[str]])
async def list_models():
    """List available models."""
    model_name = "mock-llm-model"
    return {"models": [model_name]}


@app.get("/v1/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model."""
    return {
        "model_name": "mock-llm-model",
        "max_model_len": 4096,
        "gpu_memory_usage": "0.0 GB allocated, 0.0 GB reserved (CPU only)"
    }


@app.post("/v1/completions", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text from a prompt."""
    try:
        # Log the prompt for debugging
        logger.debug(f"Received prompt: {request.prompt}")
        
        # Check for pre_prompt if provided
        if request.pre_prompt:
            logger.debug(f"Pre-prompt: {request.pre_prompt}")
            # For mock server, modify the prompt to include the pre_prompt
            combined_prompt = f"{request.pre_prompt}: {request.prompt}"
        else:
            combined_prompt = request.prompt
        
        # For simulation purposes, sometimes generate poor responses to test retry logic
        # This will randomly fail about 10% of the time to test retries
        import random
        if random.random() < 0.10:
            logger.warning("Simulating a poor response (for retry testing)")
            response_text = "<|user|>"  # Simulating a response that should trigger a retry
        else:
            # Generate a normal mock response
            response_text = generate_mock_response(combined_prompt)
        
        # Estimate token counts (very rough approximation)
        prompt_tokens = len(combined_prompt.split())
        completion_tokens = len(response_text.split())
        
        return {
            "text": response_text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_mock_response(prompt: str) -> str:
    """Generate a mock response based on the input prompt."""
    # Simple keyword-based responses
    prompt_lower = prompt.lower()
    
    if "hello" in prompt_lower or "hi" in prompt_lower:
        return "Hello! How can I help you today?"
    
    elif "weather" in prompt_lower:
        return "I'm a mock LLM and don't have access to real-time weather data. In a real implementation, I would provide current weather information."
    
    elif "time" in prompt_lower:
        return "I'm a mock LLM and don't have access to the current time. In a real implementation, I would tell you the current time."
    
    elif "name" in prompt_lower:
        return "I'm a mock voice assistant for development and testing purposes."
    
    elif "joke" in prompt_lower:
        return "Why don't scientists trust atoms? Because they make up everything!"
    
    elif "thank" in prompt_lower:
        return "You're welcome! Is there anything else I can help with?"
    
    elif "capital of france" in prompt_lower:
        return "The capital of France is Paris."
    
    elif any(word in prompt_lower for word in ["bye", "goodbye", "exit", "quit"]):
        return "Goodbye! Have a great day!"
    
    # Default response for any other prompt
    return f"This is a mock response to: '{prompt}'. In a real implementation, a language model would generate a contextually relevant response here."


def main():
    parser = argparse.ArgumentParser(description="Run a mock vLLM server for voice assistant development")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--model", type=str, default="mock-model", 
                      help="Model name (not used, just for API compatibility)")
    
    args = parser.parse_args()
    global model_path
    model_path = args.model
    
    # Log startup info
    logger.info(f"Starting mock server on {args.host}:{args.port}")
    logger.info("This is a lightweight mock server for development and testing")
    logger.info("It does NOT use any actual LLM and just returns canned responses")
    
    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()