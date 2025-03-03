import time
import logging
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from tenacity import RetryError
import ollama

logger = logging.getLogger(__name__)

class OllamaClient:
    """Utility for managing LLM API calls."""
    
    def __init__(self, model_name: str = "lama3:latest", max_retries: int = 5):
        """
        Initialize the LLM client.s
        
        Args:
            model_name: Name of the model to use (default: lama3:latest")
            max_retries: Maximum number of retry attempts
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self._client = None
        logging.basicConfig(level=logging.INFO)

    @property
    def client(self):
        """Lazy load the Ollama client."""
        if self._client is None:
            self._client = ollama.Client()
        return self._client

    @retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=10), retry=retry_if_exception_type(Exception))
    def call(
        self, 
        prompt: str, 
        temperature: float = 0.3, 
        max_tokens: int = 2048, 
        response_format: Optional[dict] = None
    ) -> str:
        """
        Call the LLM model with retries and error handling.
        
        Args:
            prompt: The input prompt for the model
            temperature: Sampling temperature (not used in Ollama, but kept for compatibility)
            max_tokens: Maximum number of tokens in the response (not used in Ollama, but kept for compatibility)
            response_format: Optional parameter to specify response format (not used in Ollama, but kept for compatibility)
            
        Returns:
            The model's response content
        """
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )
            
            logger.debug(f"LLM Response: {response}")  
            return response.get("response", "No response available.")
        
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            raise