#!/usr/bin/env python3
"""
Voice Assistant Client

This script runs on a Raspberry Pi and:
1. Listens for voice input via microphone
2. Transcribes the audio to text using Whisper
3. Sends the text to the vLLM server
4. Receives the response and converts it to speech
"""

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from typing import Optional

import requests
import sounddevice as sd
import numpy as np
import torch
from transformers import pipeline
import pyttsx3

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global variables
audio_queue = queue.Queue()
is_recording = False
speech_engine = None
transcriber = None
SERVER_URL = None


def initialize_tts():
    """Initialize text-to-speech engine."""
    global speech_engine
    speech_engine = pyttsx3.init()
    # Set properties (optional)
    speech_engine.setProperty('rate', 175)  # Speed of speech
    speech_engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
    voices = speech_engine.getProperty('voices')
    if voices:
        # Use a female voice if available
        for voice in voices:
            if "female" in voice.name.lower():
                speech_engine.setProperty('voice', voice.id)
                break
    logger.info("Text-to-speech engine initialized")


def initialize_whisper(device: str = "cpu"):
    """Initialize Whisper model for speech-to-text."""
    global transcriber
    try:
        # Use smaller model for development
        model_name = "openai/whisper-small"
        
        # Check if we're on macOS and using MPS (Metal Performance Shaders)
        if device == "mps":
            try:
                import torch
                if torch.backends.mps.is_available():
                    logger.info("Using MPS (Metal Performance Shaders) for Whisper on macOS")
                else:
                    logger.warning("MPS requested but not available, falling back to CPU")
                    device = "cpu"
            except (ImportError, AttributeError):
                logger.warning("Could not check MPS availability, falling back to CPU")
                device = "cpu"
        
        logger.info(f"Loading Whisper model: {model_name} on device: {device}")
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device
        )
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        sys.exit(1)


def audio_callback(indata, frames, time, status):
    """Callback for sounddevice to capture audio."""
    if status:
        logger.warning(f"Audio callback status: {status}")
    if is_recording:
        audio_queue.put(indata.copy())


def start_recording(sample_rate=16000):
    """Start recording audio from microphone."""
    global is_recording
    try:
        is_recording = True
        # Start streaming from microphone
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            callback=audio_callback
        )
        stream.start()
        return stream
    except Exception as e:
        logger.error(f"Failed to start recording: {e}")
        is_recording = False
        return None


def stop_recording(stream):
    """Stop recording audio."""
    global is_recording
    if stream is not None:
        stream.stop()
        stream.close()
    is_recording = False


def transcribe_audio(audio_data, sample_rate=16000):
    """Transcribe audio data to text using Whisper."""
    if transcriber is None:
        logger.error("Transcriber not initialized")
        return None

    try:
        # Convert the audio data to the format expected by Whisper
        audio = np.concatenate(audio_data)
        result = transcriber({"raw": audio, "sampling_rate": sample_rate})
        transcription = result["text"].strip()
        return transcription
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None


def send_to_llm(prompt):
    """Send prompt to LLM server and get response."""
    if not SERVER_URL:
        logger.error("Server URL not set")
        return "Server connection not configured."

    try:
        # Prepare the request
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95
        }
        
        # Send the request
        response = requests.post(
            f"{SERVER_URL}/v1/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=30
        )
        
        # Process the response
        if response.status_code == 200:
            return response.json()["text"]
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return f"Error: Failed to get response from server. Status code: {response.status_code}"
    except Exception as e:
        logger.error(f"Request error: {e}")
        return f"Error: {str(e)}"


def text_to_speech(text):
    """Convert text to speech and play it."""
    if speech_engine is None:
        logger.error("Speech engine not initialized")
        return
    
    try:
        speech_engine.say(text)
        speech_engine.runAndWait()
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")


def main_loop():
    """Main interaction loop."""
    logger.info("Starting main interaction loop")
    logger.info("Press Enter to start recording, then press Enter again to stop and process")
    
    while True:
        # Wait for user to press Enter to start recording
        input("Press Enter to start recording...")
        
        logger.info("Recording started... Press Enter to stop")
        stream = start_recording()
        audio_data = []
        
        # Record until user presses Enter again
        input()
        stop_recording(stream)
        
        # Get all audio data from the queue
        while not audio_queue.empty():
            audio_data.append(audio_queue.get())
        
        if not audio_data:
            logger.warning("No audio recorded")
            continue
        
        logger.info("Transcribing audio...")
        transcription = transcribe_audio(audio_data)
        
        if not transcription:
            logger.warning("Failed to transcribe audio")
            text_to_speech("Sorry, I couldn't understand what you said.")
            continue
        
        logger.info(f"Transcription: {transcription}")
        
        logger.info("Sending to LLM...")
        response = send_to_llm(transcription)
        logger.info(f"LLM Response: {response}")
        
        logger.info("Converting response to speech...")
        text_to_speech(response)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Voice Assistant Client")
    parser.add_argument("--server", type=str, required=True, help="URL of the vLLM server")
    parser.add_argument("--device", type=str, default="cpu", 
                      help="Device for Whisper model (cpu, cuda, mps)")
    
    args = parser.parse_args()
    global SERVER_URL
    SERVER_URL = args.server
    
    # Initialize components
    logger.info(f"Connecting to server: {SERVER_URL}")
    logger.info("Initializing text-to-speech engine...")
    initialize_tts()
    
    logger.info("Initializing Whisper model...")
    initialize_whisper(args.device)
    
    # Start main loop
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        if speech_engine:
            speech_engine.stop()
        logger.info("Voice assistant client stopped")


if __name__ == "__main__":
    main()