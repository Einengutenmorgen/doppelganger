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

            # Unlike OpenAI, Gemini doesn't use a "system" role in the same way
            # We'll prepend any system-like instructions to the user prompt
            
            formatted_prompt = prompt
            if response_format and response_format.get("type") == "json_object":
                formatted_prompt = "Please respond with a valid JSON object.\n\n" + prompt

            # Generate the response - Gemini accepts simpler input format
            response = self.model.generate_content(
                formatted_prompt,
                generation_config=generation_config
            )

            # Check if generation was successful
            if response and response.text:
                return response.text
            else:
                logger.warning("Gemini API returned empty response")
                return '{"error": "Empty response from Gemini API"}'

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise