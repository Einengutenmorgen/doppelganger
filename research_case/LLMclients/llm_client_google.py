import logging
import os
from typing import Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv

from research_case.analyzers.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)

class GeminiLLMClient(BaseLLMClient):
    """LLM client for interacting with the Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash-001"):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google API key (can be None if env variable is set)
            model_name: Name of the model to use
        """
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('GOOGLE_API_KEY')
            
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Successfully initialized Gemini model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise

    def call(self, 
             prompt: str, 
             temperature: float = 0.5, 
             max_tokens: int = 1000, 
             response_format: Optional[Dict] = None) -> str:
        """
        Call the Gemini API with the given parameters.
        
        Args:
            prompt: The input prompt text
            temperature: Controls randomness in generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            response_format: Not used for Gemini but included for interface compatibility
            
        Returns:
            Generated text string or error message
        """
        try:
            # Create the generation config - Gemini uses different parameter names
            generation_config = {
                'temperature': temperature,
                'max_output_tokens': max_tokens,
                'top_p': 0.95,
                'top_k': 40,
            }

            # Format the prompt to ensure we get a JSON response when needed
            formatted_prompt = prompt
            if response_format and response_format.get("type") == "json_object":
                # Add explicit instructions for JSON formatting
                formatted_prompt = (
                    "You must respond with a valid, properly formatted JSON object and nothing else. "
                    "Do not include markdown formatting, code blocks, or any text outside the JSON object. "
                    "Ensure all keys are properly quoted and all values are valid JSON.\n\n"
                    + prompt
                )
                
                # Log the modified prompt
                logger.debug(f"Modified prompt for JSON response: {formatted_prompt[:100]}...")

            # Generate the response
            logger.info("Sending request to Gemini API")
            response = self.model.generate_content(
                formatted_prompt,
                generation_config=generation_config
            )

            # Check if generation was successful
            if response and hasattr(response, 'text') and response.text:
                # Log a snippet of the response for debugging
                logger.debug(f"Received response from Gemini: {response.text[:100]}...")
                
                # Clean up response if JSON was requested
                if response_format and response_format.get("type") == "json_object":
                    # Try to extract JSON if wrapped in code blocks
                    text = response.text
                    
                    # Remove markdown code blocks if present
                    import re
                    json_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
                    if json_block_match:
                        logger.info("Extracted JSON from code block")
                        return json_block_match.group(1).strip()
                    
                    # Remove any non-JSON text before or after
                    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
                    if json_match:
                        logger.info("Extracted JSON object from response")
                        return json_match.group(1).strip()
                
                return response.text
            else:
                logger.warning("Gemini API returned empty response")
                return '{"error": "Empty response from Gemini API"}'

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return f'{{"error": "Error calling Gemini API: {str(e)}"}}'