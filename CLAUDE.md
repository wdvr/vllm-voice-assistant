# VLLM Voice Assistant Code Guidelines

## Project Overview
This project creates a voice assistant running on a Raspberry Pi that connects to a local Ubuntu server with NVIDIA hardware running vLLM for AI inference.

## Development Roadmap & TODOs

1. **Backend Setup** [IN PROGRESS]
   - [DONE] Install CUDA 12.8.1 and cuDNN 9.8.0 dependencies
   - [TODO] Install vLLM on Ubuntu server
   - [TODO] Download Llama 3.2 model (~8GB for 8B parameter version)
   - [TODO] Download DeepSeek model (~7GB for compatible version)
   - [TODO] Create server script with vLLM setup optimized for 2080Ti (11GB VRAM)
   - [TODO] Test inference performance and adjust quantization if needed
   - [TODO] Create systemd service for automatic server startup
   - [TODO] Add server health monitoring and automatic recovery

2. **Voice Interface**
   - [TODO] Create Python client for Raspberry Pi
   - [TODO] Implement speech-to-text using Whisper or similar
   - [TODO] Create API to send prompts to vLLM server
   - [TODO] Implement text-to-speech for responses
   - [TODO] Set up proper audio input/output on Raspberry Pi
   - [TODO] Create systemd service for automatic client startup
   - [TODO] Add reconnection logic for network interruptions

3. **Model Switching**
   - [TODO] Design model config storage format
   - [TODO] Create API endpoints for model selection
   - [TODO] Implement efficient model loading/unloading
   - [TODO] Create model performance comparison tool

4. **Always-On Functionality**
   - [TODO] Research lightweight trigger word detection
   - [TODO] Implement wake word detection ("Hey Assistant")
   - [TODO] Optimize for low power consumption
   - [TODO] Add option to disable audio feedback for quiet operation

5. **Development Tools**
   - [TODO] Create local development environment setup script
   - [TODO] Add test script to run server/client on one machine
   - [TODO] Create simulation mode for testing without hardware
   - [TODO] Set up CI/CD pipeline for testing
   - [TODO] Add debug mode with detailed logging
   - [TODO] Fix FastAPI deprecation warning: Replace @app.on_event with lifespan event handlers in server/vllm_server.py

## Build & Run Commands
- Server Setup: `pip install -r server/requirements.txt`
- Run Server: `python server/vllm_server.py --model ./models/llama3-8b --gpu-memory-utilization 0.9`
- Client Setup: `pip install -r client/requirements.txt`
- Run Client: `python client/voice_client.py --server http://SERVER_IP:8000`
- Tests: (TBD as project develops)

## Code Style Guidelines
- **Python**: Follow PEP 8 standards
- **Naming**: Use snake_case for Python variables/functions, CamelCase for classes
- **Docstrings**: Required for all functions, classes, and modules
- **Error Handling**: Use try/except blocks with specific exceptions
- **Typing**: Use type hints for all function parameters and return values
- **Comments**: Only for complex logic, prefer self-documenting code
- **Imports**: Group standard library, third-party, and local imports

## Architecture
- Server: Ubuntu with NVIDIA GPU (2080Ti, 11GB VRAM) running vLLM inference
- Client: Raspberry Pi with microphone and speaker
- Communication: (TBD as project develops)