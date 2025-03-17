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

2. **Voice Interface** [IN PROGRESS]
   - [IN PROGRESS] (#8) Create Python client for Raspberry Pi
   - [IN PROGRESS] (#9) Implement speech-to-text using Whisper or similar
   - [IN PROGRESS] (#10) Create API to send prompts to vLLM server
   - [IN PROGRESS] (#11) Implement text-to-speech for responses
   - [TODO] (#12) Set up proper audio input/output on Raspberry Pi
   - [TODO] (#13) Create systemd service for automatic client startup
   - [TODO] (#14) Add reconnection logic for network interruptions
   - [TODO] (#29) Implement voice activity detection to auto-stop recording
   - [TODO] (#30) Add support for continuous conversation mode
   - [TODO] (#31) Create more user-friendly interface for voice client
   - [TODO] (#32) Add configuration file for client settings

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

5. **Development Tools** [IN PROGRESS]
   - [DONE] Add test scripts for server unit and end-to-end testing
   - [DONE] Add development test script for interactive testing
   - [DONE] (#34) Create mock LLM server for development without GPU
   - [DONE] (#33) Add support for Metal GPU acceleration (MPS) on macOS
   - [DONE] (#35) Implement end-to-end test for voice assistant
   - [TODO] (#23) Create local development environment setup script
   - [DONE] (#24) Add test script to run server/client on one machine
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
- All Tests: `python tests/run_tests.py --all`
- Unit Tests: `python tests/run_tests.py --unit`
- End-to-End Tests: `python tests/run_tests.py --e2e`
- Interactive E2E Testing: `python tests/run_tests.py --e2e --interactive`

### Unit Testing:
- Server Unit Tests: `python tests/unit/server/test_server.py`
- Server Prompt Formatter Tests: `python tests/unit/server/test_prompt_formatter.py`
- Client Unit Tests: `python tests/unit/client/test_voice_client.py`

### Development and Testing:
- Server Dev Testing: `python scripts/dev_test.py --model ./models/phi-2 --gpu-util 0.8`
- Mock Server (CPU/macOS): `python scripts/mock_llm_server.py`
- Client Dev Testing (macOS): `python scripts/test_voice_client.py --device mps` (or `--device cpu`)
- Server E2E Tests: `python scripts/test_server_e2e.py --gpu-util 0.8`

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
- ✅ Created mock LLM server for development on macOS
- ✅ Added Metal GPU acceleration support for macOS development
- 🚧 Client implementation in progress
  - ✅ Basic speech-to-text with Whisper implemented
  - ✅ Basic text-to-speech with pyttsx3 implemented
  - ✅ API communication with server implemented
  - ✅ Created development testing framework for macOS
  - ✅ Implemented unit tests for client components
  - ❌ Proper error handling and production readiness
- ❌ Model switching functionality not yet implemented
- ❌ Always-on functionality not yet implemented
