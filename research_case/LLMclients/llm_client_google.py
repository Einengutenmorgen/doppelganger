import json
import logging
import os
from typing import Dict, List, Union

from google.generativeai import text

logger = logging.getLogger(__name__)

class GeminiLLMClient:
    """LLM client for interacting with the Gemini API."""

    def __init__(self, api_key: str):
        """Initializes the Gemini client."""
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        text.configure(api_key=api_key)  # Configure Gemini API key
        self.model_name = "models/gemini-pro" # or the most appropriate model for your use case.

    def call(self, prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
        """Calls the Gemini API with the given prompt."""
        try:

            response = text.generate_text(
                model=self.model_name,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            if response.result:
                return response.result  # Extract and return the generated text
            else:
                logger.warning(f"Gemini API returned no result. Response: {response}")
                return ""  # Or handle the empty response as needed

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise  # Re-raise the exception for handling in the calling function
