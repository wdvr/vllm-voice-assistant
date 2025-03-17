# VLLM Home Voice Assistant

This is a repo that has steps as well as the code needed to run a local AI voice assistant, backed by Llama / DeepSeek, running locally on an Ubuntu server with NVIDIA hardware, running vLLM. 
On the frontend this is using a Raspberry Pi with speaker and microphone to query the LLM.

## Getting Started

### Server Setup (Ubuntu with NVIDIA GPU)

#### Prerequisites
- Ubuntu 22.04 or later
- NVIDIA GPU with at least 11GB VRAM (optimized for RTX 2080Ti)
- NVIDIA drivers (minimum version 535)
- CUDA 12.8.1 (tested and working as of March 2025)
- cuDNN 9.8.0 (tested and working as of March 2025)
- Python 3.12 (tested and working as of March 2025)

#### Installation Steps

1. **Setup Virtual Environment**
   ```bash
   # Create a virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r server/requirements.txt
   ```

2. **Download Models**
   ```bash
   # Create model directory
   mkdir -p models
   
   # Download models (choose one that fits in your VRAM)
   # For Llama 3.2 8B
   huggingface-cli download meta-llama/Meta-Llama-3.2-8B-Instruct --local-dir ./models/llama3-8b
   
   # For DeepSeek (compatible size)
   huggingface-cli download deepseek-ai/deepseek-coder-7b-instruct-v1.5 --local-dir ./models/deepseek-coder-7b
   ```

3. **Running the vLLM Server**
   ```bash
   # Start server with Llama 3.2 (8B) model
   # Make sure your virtual environment is activated
   source venv/bin/activate
   python server/vllm_server.py --model ./models/llama3-8b --gpu-memory-utilization 0.9
   ```

4. **Setting up as a System Service (Optional)**
   ```bash
   # Edit the service file with your paths and username
   nano server/vllm-server.service
   
   # Copy to systemd directory
   sudo cp server/vllm-server.service /etc/systemd/system/
   
   # Enable and start the service
   sudo systemctl daemon-reload
   sudo systemctl enable vllm-server
   sudo systemctl start vllm-server
   
   # Check status
   sudo systemctl status vllm-server
   ```

### Client Setup for Development (macOS)

1. **Setup Development Environment**
   ```bash
   # Create a virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install Python dependencies
   pip install -r client/requirements.txt
   ```

2. **Running the Mock Server (No GPU Required)**
   ```bash
   # Start the mock LLM server in one terminal
   python scripts/mock_llm_server.py
   ```

3. **Testing the Client**
   ```bash
   # In another terminal, run the test script
   # With Metal GPU acceleration (Apple Silicon)
   python scripts/test_voice_client.py --device mps
   
   # Or with CPU (any Mac)
   python scripts/test_voice_client.py --device cpu
   ```

4. **Usage in Test Mode**
   - Type `r` to start recording your voice
   - Ask your question or give a command
   - Press Enter to stop recording
   - The system will transcribe your speech, send it to the mock server, and speak the response
   - Type `q` to quit the test

### Client Setup (Raspberry Pi)

1. **Setup Raspberry Pi OS**
   - Install Raspberry Pi OS (64-bit recommended)
   - Ensure you have Python 3.9+ installed
   - Connect a USB microphone and speakers

2. **Install Dependencies**
   ```bash
   # Install system dependencies
   sudo apt update
   sudo apt install -y python3-pip python3-dev portaudio19-dev python3-pyaudio espeak
   
   # Create a virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install Python dependencies
   cd client
   pip install -r requirements.txt
   ```

3. **Running the Client**
   ```bash
   # Make sure your virtual environment is activated
   source venv/bin/activate
   
   # Replace SERVER_IP with your Ubuntu server's IP address
   python client/voice_client.py --server http://SERVER_IP:8000
   ```

4. **Usage**
   - Press Enter to start recording your voice
   - Ask your question or give a command
   - Press Enter again to stop recording
   - The system will transcribe your speech, send it to the server, and speak the response

5. **Setting up as a System Service (Optional)**
   ```bash
   # Edit the service file with your server IP
   nano client/voice-client.service
   
   # Copy to systemd directory
   sudo cp client/voice-client.service /etc/systemd/system/
   
   # Enable and start the service
   sudo systemctl daemon-reload
   sudo systemctl enable voice-client
   sudo systemctl start voice-client
   
   # Check status
   sudo systemctl status voice-client
   ```

## Components

- **Server**: Python application using vLLM for inference
- **Client**: Raspberry Pi application with speech-to-text and text-to-speech
- **Models**: Llama 3.2 and DeepSeek

## Architecture
![Architecture Diagram (coming soon)]()

## Development

For developers who want to test or contribute to this project without setting up the full hardware environment:

1. **Local Development with GPU**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/vllm-voice-assistant.git
   cd vllm-voice-assistant
   
   # Create and activate virtual environment
   python3.12 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r server/requirements.txt
   
   # Download a smaller model for testing
   mkdir -p models
   huggingface-cli download microsoft/phi-2 --local-dir ./models/phi-2
   
   # Run the development test script
   python scripts/dev_test.py --model ./models/phi-2
   ```

2. **Development Without GPU (macOS/Windows)**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/vllm-voice-assistant.git
   cd vllm-voice-assistant
   
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # on macOS/Linux
   # or
   venv\Scripts\activate     # on Windows
   
   # Install client dependencies (includes mock server)
   pip install -r client/requirements.txt
   
   # Run the mock LLM server (no real AI, just canned responses)
   python scripts/mock_llm_server.py
   ```

3. **Testing Without Voice Hardware**
   - The `scripts/dev_test.py` script allows you to test the server with text input
   - The `scripts/test_voice_client.py` script allows you to test the client with your computer's microphone

For more detailed development instructions, see [CONTRIBUTING.md](CONTRIBUTING.md).

### Testing

The project includes several testing tools to ensure everything works correctly:

1. **Unified Test Runner**
   ```bash
   # Run all tests
   python tests/run_tests.py --all
   
   # Run just unit tests
   python tests/run_tests.py --unit
   
   # Run just end-to-end tests
   python tests/run_tests.py --e2e
   
   # Run end-to-end tests with interactive mode
   python tests/run_tests.py --e2e --interactive
   ```

2. **Unit Tests**
   ```bash
   # Run all unit tests
   python tests/run_tests.py --unit
   
   # Run specific unit tests
   python tests/unit/client/test_voice_client.py
   python tests/unit/server/test_prompt_formatter.py
   python tests/unit/server/test_server.py
   ```

3. **End-to-End Tests**
   ```bash
   # Run complete end-to-end test with mock server and client
   python tests/e2e/test_voice_assistant.py
   
   # Run server-specific end-to-end tests (requires GPU)
   python scripts/test_server_e2e.py
   ```

4. **Manual Testing with Dev Scripts**
   ```bash
   # Test vLLM server in interactive mode (requires GPU)
   python scripts/dev_test.py --model ./models/phi-2 --gpu-util 0.8
   
   # Test with mock server (no GPU required)
   python scripts/mock_llm_server.py
   
   # Test voice client on macOS
   python scripts/test_voice_client.py --device mps  # for Apple Silicon
   python scripts/test_voice_client.py --device cpu  # for any Mac
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


