# Contributing to vLLM Voice Assistant

Thank you for considering contributing to this project! This guide will help you get started with development and provide guidelines for contributing.

## Development Setup

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA support (for full testing)
- Git

### Local Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vllm-voice-assistant.git
   cd vllm-voice-assistant
   ```

2. **Set up a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Server dependencies
   pip install -r server/requirements.txt
   
   # Client dependencies (if you want to test the client)
   pip install -r client/requirements.txt
   ```

4. **Download a smaller model for testing**
   For development, you can use a smaller model that fits in less VRAM:
   ```bash
   mkdir -p models
   # Example using a smaller model
   huggingface-cli download microsoft/phi-2 --local-dir ./models/phi-2
   ```

5. **Run the development test script**
   ```bash
   python scripts/dev_test.py --model ./models/phi-2
   ```

### Running Components Separately

#### Server
```bash
python server/vllm_server.py --model ./models/your-model --gpu-memory-utilization 0.7
```

#### Client
```bash
# For development, you can run the client with a local server
python client/voice_client.py --server http://localhost:8000
```

## Development Guidelines

### Code Style

- Follow PEP 8 standards
- Use type hints for all function parameters and return values
- Write docstrings for all functions, classes, and modules
- Use snake_case for variables and functions, CamelCase for classes

### Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with descriptive commit messages
3. Add or update tests as necessary
4. Update documentation to reflect any changes
5. Submit a pull request with a clear description of the changes

### Testing

Before submitting a PR, please:
1. Test your changes in a development environment
2. Run the dev_test.py script to ensure the server functions correctly
3. If you made changes to the client, test it with a local server

## Project Structure

- `/server`: vLLM server implementation
- `/client`: Raspberry Pi client implementation
- `/models`: Directory for storing LLM models
- `/scripts`: Utility scripts for development and testing
- `/docs`: Documentation

## Getting Help

If you have questions or need help with the project:
1. Check existing documentation
2. Open an issue on GitHub
3. Contact the project maintainers