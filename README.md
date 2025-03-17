# VLLM Home Voice Assistant

This is a repo that has steps as well as the code needed to run a local AI voice assistant, backed by Llama / DeepSeek, running locally on an Ubuntu server with NVIDIA hardware, running vLLM. 
On the frontend this is using a Raspberry Pi with speaker and microphone to query the LLM.

## Getting Started

### Server Setup (Ubuntu with NVIDIA GPU)

#### Prerequisites
- Ubuntu 22.04 or later
- NVIDIA GPU with at least 11GB VRAM (optimized for RTX 2080Ti)
- NVIDIA drivers (minimum version 535)
- CUDA 12.4 (latest as of March 2025)
- cuDNN 9.0 (latest as of March 2025)

#### Installation Steps

1. **Install NVIDIA Drivers, CUDA, and cuDNN**
   ```bash
   # Install NVIDIA drivers
   sudo apt update
   sudo apt install nvidia-driver-535
   
   # Install CUDA 12.4
   wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
   sudo sh cuda_12.4.0_550.54.14_linux.run
   
   # Set up environment variables
   echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   
   # Install cuDNN (download from NVIDIA requires an account)
   # After downloading the .deb file from NVIDIA Developer site:
   sudo dpkg -i cudnn-local-repo-ubuntu2204-9.0.0.x_1.0-1_amd64.deb
   sudo cp /var/cudnn-local-repo-*/cudnn-*-keyring.gpg /usr/share/keyrings/
   sudo apt update
   sudo apt install libcudnn9
   ```

2. **Install Python and Dependencies**
   ```bash
   sudo apt install python3.10-dev python3-pip
   python3 -m pip install --upgrade pip
   ```

3. **Install vLLM**
   ```bash
   pip install vllm
   ```

4. **Download Models**
   ```bash
   # Create model directory
   mkdir -p models
   
   # Download models (choose one that fits in your VRAM)
   # For Llama 3.2 8B
   huggingface-cli download meta-llama/Meta-Llama-3.2-8B-Instruct --local-dir ./models/llama3-8b
   
   # For DeepSeek (compatible size)
   huggingface-cli download deepseek-ai/deepseek-coder-7b-instruct-v1.5 --local-dir ./models/deepseek-coder-7b
   ```

5. **Running the vLLM Server**
   ```bash
   # Start server with Llama 3.2 (8B) model
   python3 server/vllm_server.py --model ./models/llama3-8b --gpu-memory-utilization 0.9
   ```

6. **Setting up as a System Service**
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
   
   # Install Python dependencies
   cd client
   pip3 install -r requirements.txt
   ```

3. **Running the Client**
   ```bash
   # Replace SERVER_IP with your Ubuntu server's IP address
   python3 voice_client.py --server http://SERVER_IP:8000
   ```

4. **Usage**
   - Press Enter to start recording your voice
   - Ask your question or give a command
   - Press Enter again to stop recording
   - The system will transcribe your speech, send it to the server, and speak the response

5. **Setting up as a System Service**
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

1. **Local Development**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/vllm-voice-assistant.git
   cd vllm-voice-assistant
   
   # Install dependencies
   pip install -r server/requirements.txt
   
   # Download a smaller model for testing
   mkdir -p models
   huggingface-cli download microsoft/phi-2 --local-dir ./models/phi-2
   
   # Run the development test script
   python scripts/dev_test.py --model ./models/phi-2
   ```

2. **Testing Without Voice Hardware**
   The `scripts/dev_test.py` script allows you to test the system with text input instead of voice, making it easier to develop and test without Raspberry Pi hardware.

For more detailed development instructions, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


