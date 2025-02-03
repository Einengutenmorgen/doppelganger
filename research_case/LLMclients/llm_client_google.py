import logging
import os
import google.generativeai as genai
from typing import Optional

logger = logging.getLogger(__name__)

class GeminiLLMClient:
    """LLM client for interacting with the Gemini API."""

    def __init__(self, api_key: str):
        """Initializes the Gemini client."""
        if api_key == None:
            load_dotenv()
            api_key=os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Successfully initialized Gemini model")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise

    def call(self, 
             prompt: str, 
             temperature: float = 0.1, 
             max_tokens: int = 500) -> Optional[str]:
        """
        Calls the Gemini API with the given prompt.
        
        Args:
            prompt: The input prompt text
            temperature: Controls randomness in generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text string or None if generation fails
        """
        try:
            # Create the generation config
            generation_config = {
                'temperature': temperature,
                'max_output_tokens': max_tokens,
                'top_p': 0.95,
                'top_k': 40,
            }

            # Generate the response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Check if generation was successful
            if response and response.text:
                return response.text
            else:
                logger.warning("Gemini API returned empty response")
                return None

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise