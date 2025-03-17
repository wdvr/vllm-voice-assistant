# Voice Assistant Client

This directory contains the client code for the voice assistant that runs on a Raspberry Pi. For development purposes, it can also be tested on macOS using the built-in microphone.

## Overview

The client performs these main functions:
1. Listens for voice input via microphone
2. Transcribes the audio to text using Whisper
3. Sends the text to the vLLM server
4. Receives the response and converts it to speech

## Setup for macOS Development

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the mock LLM server in one terminal:
   ```
   python scripts/mock_llm_server.py
   ```

4. Run the test script in another terminal:
   ```
   python scripts/test_voice_client.py --device mps  # Use Metal GPU acceleration if available
   ```

   Or use CPU (slower but more compatible):
   ```
   python scripts/test_voice_client.py --device cpu
   ```

## Unit Tests

Run unit tests with:
```
python client/run_unit_tests.py
```

## Notes for macOS Development

- The client uses `pyttsx3` for text-to-speech, which requires `pyobjc` on macOS
- For Whisper speech recognition, you can use Metal Performance Shaders (MPS) on Apple Silicon for better performance (use `--device mps`)
- The `-r` command initiates recording with the microphone
- For audio input/output on macOS, you may need to grant terminal permissions in System Preferences > Security & Privacy > Microphone
- The mock LLM server provides canned responses for testing without a GPU
- Unit tests use mocks to avoid loading the actual Whisper model, making them run quickly

## Next Steps (TODO)

- [ ] (#20) Implement wake word detection ("Hey Assistant")
- [ ] (#12) Set up proper audio input/output on Raspberry Pi
- [ ] (#31) Create a more user-friendly interface
- [ ] (#30) Add support for continuous conversation mode
- [ ] (#29) Implement voice activity detection to auto-stop recording
- [ ] (#32) Add configuration file for client settings