# VLLM Voice Assistant Code Guidelines

## Project Overview
This project creates a voice assistant running on a Raspberry Pi that connects to a local Ubuntu server with NVIDIA hardware running vLLM for AI inference.

## Development Roadmap & TODOs

1. **Backend Setup** [IN PROGRESS]
   - [DONE] Install CUDA 12.8.1 and cuDNN 9.8.0 dependencies
   - [DONE] Create vLLM server base implementation with model-specific prompt formatting
   - [DONE] Create testing infrastructure for server
   - [TODO] (#1) Install vLLM on Ubuntu server
   - [TODO] (#2) Download Llama 3.2 model (~8GB for 8B parameter version)
   - [TODO] (#3) Download DeepSeek model (~7GB for compatible version)
   - [TODO] (#4) Create server script with vLLM setup optimized for 2080Ti (11GB VRAM)
   - [TODO] (#5) Test inference performance and adjust quantization if needed
   - [TODO] (#6) Create systemd service for automatic server startup
   - [TODO] (#7) Add server health monitoring and automatic recovery

2. **Voice Interface**
   - [TODO] (#8) Create Python client for Raspberry Pi
   - [TODO] (#9) Implement speech-to-text using Whisper or similar
   - [TODO] (#10) Create API to send prompts to vLLM server
   - [TODO] (#11) Implement text-to-speech for responses
   - [TODO] (#12) Set up proper audio input/output on Raspberry Pi
   - [TODO] (#13) Create systemd service for automatic client startup
   - [TODO] (#14) Add reconnection logic for network interruptions

3. **Model Switching**
   - [TODO] (#15) Design model config storage format
   - [TODO] (#16) Create API endpoints for model selection
   - [TODO] (#17) Implement efficient model loading/unloading
   - [TODO] (#18) Create model performance comparison tool

4. **Always-On Functionality**
   - [TODO] (#19) Research lightweight trigger word detection
   - [TODO] (#20) Implement wake word detection ("Hey Assistant")
   - [TODO] (#21) Optimize for low power consumption
   - [TODO] (#22) Add option to disable audio feedback for quiet operation

5. **Development Tools**
   - [DONE] Add test scripts for server unit and end-to-end testing
   - [DONE] Add development test script for interactive testing
   - [TODO] (#23) Create local development environment setup script
   - [TODO] (#24) Add test script to run server/client on one machine
   - [TODO] (#25) Create simulation mode for testing without hardware
   - [TODO] (#26) Set up CI/CD pipeline for testing
   - [TODO] (#27) Add debug mode with detailed logging
   - [TODO] (#28) Fix FastAPI deprecation warning: Replace @app.on_event with lifespan event handlers in server/vllm_server.py

## Build & Run Commands
- Server Setup: `pip install -r server/requirements.txt`
- Run Server: `python server/vllm_server.py --model ./models/llama3-8b --gpu-memory-utilization 0.9`
- Client Setup: `pip install -r client/requirements.txt`
- Run Client: `python client/voice_client.py --server http://SERVER_IP:8000`

## Testing Commands
- Unit Tests: `python server/run_unit_tests.py`
- End-to-End Tests: `python scripts/test_server_e2e.py --gpu-util 0.8`
- Dev Testing: `python scripts/dev_test.py --model ./models/phi-2 --gpu-util 0.8`

## Code Style Guidelines
- **Python**: Follow PEP 8 standards
- **Naming**: Use snake_case for Python variables/functions, CamelCase for classes
- **Docstrings**: Required for all functions, classes, and modules
- **Error Handling**: Use try/except blocks with specific exceptions
- **Typing**: Use type hints for all function parameters and return values
- **Comments**: Only for complex logic, prefer self-documenting code
- **Imports**: Group standard library, third-party, and local imports

## Development Workflow
- For each new TODO item, create a corresponding GitHub issue
- Reference the issue number in the TODO comment, e.g., `[TODO] (#42) Implement feature X`
- When implementing, include the issue number in commit messages using `Fixes #42` or `Closes #42`
- Label issues appropriately: backend, voice, model-switching, always-on, devtools, bug

## Architecture
- Server: Ubuntu with NVIDIA GPU (2080Ti, 11GB VRAM) running vLLM inference
- Client: Raspberry Pi with microphone and speaker
- Communication: JSON API over HTTP/HTTPS
  - /v1/completions: Generate responses from prompts
  - /v1/models: List available models
  - /v1/model/info: Get information about the loaded model

## Implementation Progress (March 2025)
- ✅ Created vLLM server with model-specific prompt formatting
- ✅ Built comprehensive testing suite
- ✅ Fixed model compatibility issues with various LLM models (Phi, Llama, DeepSeek)
- ❌ Client implementation not yet started
- ❌ Model switching functionality not yet implemented
- ❌ Always-on functionality not yet implemented
