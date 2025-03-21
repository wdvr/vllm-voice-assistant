#!/usr/bin/env python3
"""
Development Client Test Script

This script allows testing the client's connection to the vLLM server
by sending a text prompt directly without the voice interface.
"""

import argparse
import json
import logging
import sys
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def send_query(server_url: str, prompt: str, pre_prompt: Optional[str] = None, max_tokens: int = 64, debug: bool = False):
    """Send a text query to the LLM server and return the response."""
    try:
        # Ensure server URL has the correct format
        if not server_url.startswith("http://") and not server_url.startswith("https://"):
            server_url = f"http://{server_url}"
        
        # Prepare the request
        headers = {"Content-Type": "application/json"}
        
        # Prepare the data with the prompt and optional pre_prompt
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.95
        }
        
        # Add pre_prompt if provided
        if pre_prompt:
            data["pre_prompt"] = pre_prompt
        
        endpoint = f"{server_url}/v1/completions"
        logger.info(f"Sending request to {endpoint}")
        
        # Send the request
        response = requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(data),
            timeout=30
        )
        
        # Process the response
        if response.status_code == 200:
            json_response = response.json()
            
            # Always log the full response in debug mode
            if debug:
                logger.info(f"Full API response: {json.dumps(json_response, indent=2)}")
            
            # Extract text from the response based on the server implementation
            raw_text = None
            if "text" in json_response:
                # Standard vLLM server response format
                raw_text = json_response["text"]
            elif "choices" in json_response and len(json_response["choices"]) > 0:
                # OpenAI-style API format
                raw_text = json_response["choices"][0]["text"]
            else:
                # Log the full response for debugging
                logger.error(f"Unexpected API response format: {json_response}")
                return f"Error: Unexpected API response format. Server returned: {json_response}"
            
            # Clean up the response - extract just the first response
            if raw_text:
                if debug:
                    logger.debug(f"Original response: {repr(raw_text)}")
                
                # Check for empty or very short responses
                if not raw_text.strip():
                    return "No response received from the model."
                
                # Check if we only got back the input prompt or part of it
                if raw_text.strip().startswith("<|user|>") or raw_text.strip().startswith("<|human|>"):
                    # Try to find assistant response after the user part
                    assistant_markers = ["<|assistant|>", "<|bot|>", "assistant:", "Assistant:"]
                    for marker in assistant_markers:
                        if marker in raw_text:
                            assistant_part = raw_text.split(marker, 1)[1].strip()
                            if assistant_part:
                                if debug:
                                    logger.debug(f"Found assistant response after marker: {repr(assistant_part)}")
                                raw_text = assistant_part
                                break
                    else:
                        # No assistant part found
                        return "Model did not generate a response."
                
                # Remove conversation markers
                cleaned_text = raw_text
                
                # Check for conversation markers and truncate if found
                conversation_markers = ["<|user|>", "<|human|>", "\nUser:", "\nHuman:", "<|assistant|>", 
                                       "<|end|>", "<|bot|>", "\nAssistant:"]
                for marker in conversation_markers:
                    pos = cleaned_text.find(marker)
                    if pos > 0:  # Only truncate if marker is not at the beginning
                        cleaned_text = cleaned_text[:pos].strip()
                
                # If cleaned text starts with a special token, remove it
                special_tokens = ["<s>", "<|startoftext|>", "<|im_start|>"]
                for token in special_tokens:
                    if cleaned_text.startswith(token):
                        cleaned_text = cleaned_text[len(token):].strip()
                
                # Get just the meaningful part of the response
                lines = [line for line in cleaned_text.split('\n') if line.strip()]
                if not lines:
                    return "No meaningful response received."
                
                # Handle cases where the first few lines might be part of the chat template
                if len(lines) > 1:
                    if len(lines[0]) < 10 and len(lines) > 1:
                        # Very short first line might be an interjection or partial response
                        first_response = " ".join([line.strip() for line in lines[:2]])
                    else:
                        first_response = lines[0].strip()
                else:
                    first_response = cleaned_text.strip()
                
                # Final check to handle the case where we might still have the original prompt
                if prompt in first_response and len(first_response) > len(prompt) + 10:
                    # Try to extract the part after the prompt
                    first_response = first_response.split(prompt, 1)[1].strip()
                
                if debug:
                    logger.debug(f"Cleaned response: {repr(first_response)}")
                
                return first_response
            
            return raw_text
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return f"Error: Failed to get response from server. Status code: {response.status_code}"
    except Exception as e:
        logger.error(f"Request error: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return f"Error: {str(e)}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test the vLLM server with a text prompt")
    parser.add_argument("--server", type=str, required=True, 
                        help="Server URL (e.g., '192.168.0.5:8000')")
    parser.add_argument("--prompt", type=str,
                        help="Text prompt to send to the server")
    parser.add_argument("--pre-prompt", type=str,
                        help="Optional instruction prefix (e.g., 'explain like I'm 5')")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Maximum number of tokens to generate (default: 64)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with detailed logging")
    parser.add_argument("--test-connection", action="store_true",
                        help="Test server connection without sending a prompt")
    
    args = parser.parse_args()
    
    # Set log level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Ensure server URL has the correct format
    server_url = args.server
    if not server_url.startswith("http://") and not server_url.startswith("https://"):
        server_url = f"http://{server_url}"
    
    try:
        # Always test connection first 
        logger.info(f"Testing connection to server: {server_url}")
        try:
            test_url = f"{server_url}/v1/models"
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                logger.info(f"Connection successful! Available models: {models}")
                
                # Test more detailed model info if available
                try:
                    info_response = requests.get(f"{server_url}/v1/model/info", timeout=5)
                    if info_response.status_code == 200:
                        model_info = info_response.json()
                        logger.info(f"Model info: {model_info}")
                except Exception:
                    # Not critical if this fails
                    pass
            else:
                logger.error(f"Connection test failed: {response.status_code} - {response.text}")
                if args.test_connection:
                    return
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            if args.test_connection:
                return
        
        # Exit if only testing connection
        if args.test_connection:
            return
        
        # Check if prompt is provided
        if not args.prompt:
            logger.error("No prompt provided. Use --prompt to specify a prompt.")
            return
        
        # Send the query
        logger.info(f"Sending prompt to server: {args.prompt}")
        if args.pre_prompt:
            logger.info(f"With pre-prompt: {args.pre_prompt}")
        response = send_query(
            server_url=server_url, 
            prompt=args.prompt, 
            pre_prompt=args.pre_prompt,
            max_tokens=args.max_tokens, 
            debug=args.debug
        )
        
        # Display the response
        print("\n--- LLM Response ---")
        print(response)
        print("-------------------")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()