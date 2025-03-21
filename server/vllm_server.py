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
from contextlib import asynccontextmanager

from server.prompt_formatter import formatter


# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global variables
llm = None
model_path = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Replaces deprecated @app.on_event("startup") and @app.on_event("shutdown").
    """
    # Startup: Initialize the LLM
    logger.info("Starting up vLLM server...")
    yield
    # Shutdown: Add cleanup code here if needed
    logger.info("Shutting down vLLM server...")


# Initialize FastAPI
app = FastAPI(title="Voice Assistant vLLM API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
llm = None
model_path = None
args = None

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
    if llm is None:
        return {"models": []}
    # Get the model name from the llm object
    try:
        # Try the newer vLLM API structure first
        model_name = llm.get_model_config().model_id
    except AttributeError:
        try:
            # Fallback to older vLLM API structure 
            model_name = llm.model_config.model
        except AttributeError:
            # Last resort, use a generic name
            model_name = "loaded-model"
    
    return {"models": [model_name]}


@app.get("/v1/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model."""
    if llm is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    import torch
    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    
    # Get the model name using the same logic as list_models
    try:
        # Try the newer vLLM API structure first
        model_name = llm.get_model_config().model_id
    except AttributeError:
        try:
            # Fallback to older vLLM API structure 
            model_name = llm.model_config.model
        except AttributeError:
            # Last resort, use a generic name
            model_name = "loaded-model"
    
    # Get max context length
    try:
        # Try newer vLLM API structure
        max_len = llm.get_model_config().max_model_len
    except AttributeError:
        try:
            # Fallback to older structure
            max_len = llm.llm_engine.model_config.max_model_len
        except AttributeError:
            # Default value
            max_len = 4096
    
    return {
        "model_name": model_name,
        "max_model_len": max_len,
        "gpu_memory_usage": f"{gpu_memory_allocated:.2f} GB allocated, {gpu_memory_reserved:.2f} GB reserved"
    }


@app.post("/v1/completions", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text from a prompt."""
    global model_path
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format the prompt based on the model type and pre_prompt if provided
        formatted_prompt = formatter.format_prompt(
            prompt=request.prompt,
            model_path=model_path,
            pre_prompt=request.pre_prompt
        )
        logger.debug(f"Original prompt: {request.prompt}")
        if request.pre_prompt:
            logger.debug(f"Pre-prompt: {request.pre_prompt}")
        logger.debug(f"Formatted prompt: {formatted_prompt}")
        
        # Use lower temperature and fewer tokens for more concise, predictable responses
        max_tokens = min(request.max_tokens, 64) if request.max_tokens > 64 else request.max_tokens
        temperature = min(request.temperature, 0.3) if request.temperature > 0.3 else request.temperature
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=request.top_p,
            max_tokens=max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
        )
        
        outputs = llm.generate(formatted_prompt, sampling_params)
        raw_text = outputs[0].outputs[0].text
        
        # Parse the response to clean it up based on model type
        generated_text = formatter.parse_response(raw_text, model_path)
        
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


def download_model(model_name: str, quantization: Optional[str] = None) -> tuple:
    """Download a model from Hugging Face if it doesn't exist locally."""
    from huggingface_hub import snapshot_download, HfApi
    import os

    # Map of supported models and their HF repos
    model_repos = {
        # Gated models (require Hugging Face login)
        "llama3-8b": "meta-llama/Meta-Llama-3-8B",
        "llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        
        # Open access models
        "phi-2": "microsoft/phi-2",
        "deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-base",
        "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-base"
    }
    
    # Quantized model versions for memory efficiency (optional)
    quantized_model_repos = {
        # GPTQ versions
        "deepseek-coder-6.7b": "TheBloke/deepseek-coder-6.7B-base-GPTQ",
        "llama3-8b": "TheBloke/Meta-Llama-3-8B-GPTQ"
    }
    
    # Check if we're looking for a quantized version
    if quantization == "gptq" and model_name in quantized_model_repos:
        logger.info(f"Using quantized GPTQ version of {model_name}")
        repo_id = quantized_model_repos[model_name]
        model_dir = f"./models/{model_name}-gptq"
    else:
        # Regular model repository
        if model_name not in model_repos:
            logger.error(f"Model {model_name} not found in supported models list")
            return False, None
        repo_id = model_repos[model_name]
        model_dir = f"./models/{model_name}"
    
    # Check if model is gated (requires login)
    gated_models = ["llama3-8b", "llama3-8b-instruct"]
    if model_name in gated_models:
        try:
            # Check if user is logged in
            api = HfApi()
            try:
                token_present = api.whoami()
                logger.info("Hugging Face authentication token found")
            except Exception:
                logger.error(
                    f"Model {model_name} requires Hugging Face authentication.\n"
                    "Please run 'huggingface-cli login' and follow the instructions,\n"
                    "or download the model manually from https://huggingface.co/"
                )
                return False
        except Exception as e:
            logger.error(f"Error checking authentication: {e}")
            return False
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs("./models", exist_ok=True)
        
        # Create model-specific directory
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Downloading model from Hugging Face ({repo_id})...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logger.info(f"Successfully downloaded model to {model_dir}")
        return True, model_dir  # Return success and the directory path
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        return False, None


def load_model(model_path_arg: str, gpu_mem_utilization: float, quantization: Optional[str] = None, 
              dtype: Optional[object] = None, max_model_len: Optional[int] = None):
    """Load the LLM model."""
    global llm, model_path, args
    
    # Set global model path for use in prompt formatting
    model_path = model_path_arg
    
    # Check if model exists locally, if not try to download it
    if not os.path.exists(model_path):
        # Check if it's a predefined model name rather than a path
        model_name = os.path.basename(model_path)
        
        # List of supported models for auto-download
        supported_models = ["llama3-8b", "llama3-8b-instruct", "phi-2", 
                           "deepseek-coder-6.7b", "deepseek-coder-1.3b"]
        
        if model_name in supported_models:
            logger.info(f"Model {model_name} not found locally. Attempting to download...")
            success, downloaded_path = download_model(model_name, quantization)
            if success:
                model_path = downloaded_path
                logger.info(f"Model downloaded successfully. Using path: {model_path}")
            else:
                logger.error(f"Failed to download model {model_name}")
                return False
        else:
            logger.error(f"Model path {model_path} does not exist and is not a recognized model name")
            logger.error(f"Supported models: {', '.join(supported_models)}")
            return False
    
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
        
        # Detect model type for logging purposes
        model_type = formatter.detect_model_type(model_path)
        logger.info(f"Detected model type: {model_type}")
        logger.info(f"Using appropriate prompt templates for {model_type} models")
            
        # Add dtype to kwargs if specified
        model_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_mem_utilization,
            **quantization_kwargs
        }
        
        # Add dtype if specified
        if dtype is not None:
            model_kwargs["dtype"] = dtype
            logger.info(f"Using dtype: {dtype}")
            
        # Add max_model_len if specified 
        max_model_len = getattr(args, 'max_model_len', None)
        if max_model_len is not None:
            model_kwargs["max_model_len"] = max_model_len
            logger.info(f"Using max_model_len: {max_model_len}")
            
        llm = LLM(**model_kwargs)
        logger.info(f"Model loaded: {model_path}")
        
        # Log max sequence length safely
        try:
            # Try newer vLLM API structure
            max_len = llm.get_model_config().max_model_len
            logger.info(f"Max sequence length: {max_len}")
        except AttributeError:
            try:
                # Fallback to older structure
                max_len = llm.llm_engine.model_config.max_model_len
                logger.info(f"Max sequence length: {max_len}")
            except AttributeError:
                logger.info("Could not determine max sequence length")
        
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def main():
    global args
    parser = argparse.ArgumentParser(description="Run a vLLM server for the voice assistant")
    parser.add_argument("--model", type=str, required=True, help="Path to the model or model name")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                      help="Fraction of GPU memory to use (0.0 to 1.0)")
    parser.add_argument("--quantization", type=str, choices=["awq", "gptq", "squeezellm"], 
                      help="Quantization method to use (for large models on limited VRAM)")
    parser.add_argument("--dtype", type=str, choices=["float16", "half", "bfloat16", "bf16", "float", "float32"],
                      help="Data type for model weights (e.g., float16, bfloat16)")
    parser.add_argument("--max-model-len", type=int, default=None,
                      help="Maximum sequence length for the model context window")
    
    args = parser.parse_args()
    
    # Load the model
    dtype = None
    if args.dtype:
        if args.dtype in ["float16", "half"]:
            import torch
            dtype = torch.float16
        elif args.dtype in ["bfloat16", "bf16"]:
            import torch
            dtype = torch.bfloat16
        elif args.dtype in ["float", "float32"]:
            import torch
            dtype = torch.float32
    
    if not load_model(args.model, args.gpu_memory_utilization, args.quantization, 
                     dtype=dtype, max_model_len=args.max_model_len):
        logger.error("Failed to load model. Exiting.")
        return
    
    # Run the server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()