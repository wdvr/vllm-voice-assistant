#!/usr/bin/env python3
"""
Unit Tests for Voice Client

This script runs unit tests for the voice client components.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json

# Get the project root directory and add it to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, root_dir)

# Import numpy after setting up the path
import numpy as np

# Import the client module
sys.path.append(root_dir)
from client.voice_client import (
    initialize_tts,
    initialize_whisper,
    send_to_llm,
    transcribe_audio,
    text_to_speech
)


class TestVoiceClient(unittest.TestCase):
    """Test cases for voice client components."""
    
    def setUp(self):
        # Patch global variables to prevent actual model loading
        import client.voice_client
        self._original_transcriber = client.voice_client.transcriber
        self._original_speech_engine = client.voice_client.speech_engine
        
    def tearDown(self):
        # Restore global variables
        import client.voice_client
        client.voice_client.transcriber = self._original_transcriber
        client.voice_client.speech_engine = self._original_speech_engine
    
    @patch('client.voice_client.pyttsx3.init')
    def test_initialize_tts(self, mock_init):
        """Test text-to-speech initialization."""
        # Setup mock
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        mock_engine.getProperty.return_value = [MagicMock(name='female voice', id='voice_id')]
        
        # Call the function
        initialize_tts()
        
        # Verify the function worked correctly
        mock_init.assert_called_once()
        mock_engine.setProperty.assert_any_call('rate', 175)
        mock_engine.setProperty.assert_any_call('volume', 1.0)
        mock_engine.getProperty.assert_called_with('voices')
    
    @patch('client.voice_client.pipeline')
    def test_initialize_whisper(self, mock_pipeline):
        """Test Whisper model initialization."""
        # Setup mock
        mock_pipeline.return_value = MagicMock()
        
        # Call the function
        initialize_whisper(device="cpu")
        
        # Verify the function worked correctly
        mock_pipeline.assert_called_with(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device="cpu"
        )
    
    @patch('requests.post')
    def test_send_to_llm(self, mock_post):
        """Test sending prompts to LLM server."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Mock response"}
        mock_post.return_value = mock_response
        
        # Set global variable
        import client.voice_client
        client.voice_client.SERVER_URL = "http://mock-server:8000"
        
        # Call the function
        result = send_to_llm("Test prompt")
        
        # Verify the function worked correctly
        self.assertEqual(result, "Mock response")
        mock_post.assert_called_with(
            "http://mock-server:8000/v1/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "prompt": "Test prompt",
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.95
            }),
            timeout=30
        )
    
    def test_transcribe_audio(self):
        """Test audio transcription."""
        # Setup mock
        import client.voice_client
        client.voice_client.transcriber = MagicMock()
        client.voice_client.transcriber.return_value = {"text": "Hello world"}
        
        # Create sample audio data
        audio_data = [np.zeros((1000, 1), dtype=np.float32)]
        
        # Call the function
        result = transcribe_audio(audio_data)
        
        # Verify the function worked correctly
        self.assertEqual(result, "Hello world")
        client.voice_client.transcriber.assert_called_once()
        
    def test_transcribe_audio_multi_channel(self):
        """Test audio transcription with multi-channel audio."""
        # Setup mock
        import client.voice_client
        client.voice_client.transcriber = MagicMock()
        client.voice_client.transcriber.return_value = {"text": "Hello world"}
        
        # Create sample multi-channel audio data (stereo)
        audio_data = [np.ones((1000, 2), dtype=np.float32)]
        
        # Call the function
        result = transcribe_audio(audio_data)
        
        # Verify the function worked correctly
        self.assertEqual(result, "Hello world")
        
        # Verify that the transcriber was called with flattened single-channel audio
        args, kwargs = client.voice_client.transcriber.call_args
        processed_audio = args[0]["raw"]
        
        # Check that audio is 1D (single channel)
        self.assertEqual(len(processed_audio.shape), 1, "Audio should be single-channel")
        self.assertEqual(processed_audio.shape[0], 1000, "Audio length should be preserved")
    
    def test_text_to_speech(self):
        """Test text-to-speech conversion."""
        # Setup mock
        mock_engine = MagicMock()
        
        # Set the global variable
        import client.voice_client
        client.voice_client.speech_engine = mock_engine
        
        # Call the function
        text_to_speech("Hello world")
        
        # Verify the function worked correctly
        mock_engine.say.assert_called_with("Hello world")
        mock_engine.runAndWait.assert_called_once()


if __name__ == "__main__":
    unittest.main()