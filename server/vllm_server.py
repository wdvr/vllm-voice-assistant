#!/usr/bin/env python3
"""
vLLM Server for Voice Assistant

This script runs a FastAPI server that serves LLM models using vLLM.
Optimized for RTX 2080Ti with 11GB VRAM.
"""

import argparse
import logging
import os
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Voice Assistant vLLM API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM instance
llm = None


class GenerationRequest(BaseModel):
    """Request body for text generation."""
    prompt: str = Field(..., description="The prompt to generate text from")
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


@app.on_event("startup")
async def startup_event():
    """Initialize the LLM on server startup."""
    logger.info("Starting up vLLM server...")


@app.get("/v1/models", response_model=Dict[str, List[str]])
async def list_models():
    """List available models."""
    if llm is None:
        return {"models": []}
    return {"models": [llm.model_config.model]}


@app.get("/v1/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model."""
    if llm is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    import torch
    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    
    return {
        "model_name": llm.model_config.model,
        "max_model_len": llm.llm_engine.model_config.max_model_len,
        "gpu_memory_usage": f"{gpu_memory_allocated:.2f} GB allocated, {gpu_memory_reserved:.2f} GB reserved"
    }


@app.post("/v1/completions", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text from a prompt."""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
        )
        
        outputs = llm.generate(request.prompt, sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # Estimate token counts (actual counts would require tokenizer)
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(generated_text.split())
        
        return {
            "text": generated_text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def load_model(model_path: str, gpu_mem_utilization: float, quantization: Optional[str] = None):
    """Load the LLM model."""
    global llm
    try:
        tensor_parallel_size = 1  # Single GPU setup
        
        quantization_kwargs = {}
        if quantization:
            if quantization == "awq":
                quantization_kwargs = {"quantization": "awq"}
            elif quantization == "squeezellm":
                quantization_kwargs = {"quantization": "squeezellm"}
            elif quantization == "gptq":
                quantization_kwargs = {"quantization": "gptq"}
            
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_mem_utilization,
            **quantization_kwargs
        )
        logger.info(f"Model loaded: {model_path}")
        logger.info(f"Max sequence length: {llm.llm_engine.model_config.max_model_len}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run a vLLM server for the voice assistant")
    parser.add_argument("--model", type=str, required=True, help="Path to the model or model name")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                      help="Fraction of GPU memory to use (0.0 to 1.0)")
    parser.add_argument("--quantization", type=str, choices=["awq", "gptq", "squeezellm"], 
                      help="Quantization method to use (if supported by the model)")
    
    args = parser.parse_args()
    
    # Load the model
    if not load_model(args.model, args.gpu_memory_utilization, args.quantization):
        logger.error("Failed to load model. Exiting.")
        return
    
    # Run the server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()