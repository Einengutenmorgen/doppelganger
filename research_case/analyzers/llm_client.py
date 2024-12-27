import time
import logging
from typing import Dict, List, Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from tenacity import RetryError
#import requests
#from requests import TypeError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    """Utility for managing LLM API calls."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o", max_retries: int = 5):
        """
        Initialize the LLM client.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the model to use
            max_retries: Maximum number of retry attempts
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries

    #@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=10),retry=retry_if_exception_type(TypeError))
    def call(self, prompt: str, temperature: float = 0.5, max_tokens: int = 1000) -> str:
        """
        Call the LLM model with retries and error handling.
        
        Args:
            prompt: The input prompt for the model
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            The model's response content
        """
        try:
            response =  self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            logger.debug(f"LLM Response: {response}")  # Log the raw response
            return response.choices[0].message.content
        
        except TypeError as te:
            logger.error(f"TypeError encountered: {te}", exc_info=True)
            return self._get_default_evaluation()
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            raise