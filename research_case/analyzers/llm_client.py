import time
import logging
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""
    
    @abstractmethod
    def call(self, 
             prompt: str, 
             temperature: float = 0.5, 
             max_tokens: int = 1000, 
             response_format: Optional[Dict] = None) -> str:
        """
        Call the LLM model with the given parameters.
        
        Args:
            prompt: The input prompt for the model
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens in the response
            response_format: Optional parameter to specify response format
            
        Returns:
            The model's response content
        """
        pass

class LLMClient(BaseLLMClient):
    """OpenAI-specific implementation of LLM client."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o", max_retries: int = 5):
        """
        Initialize the OpenAI LLM client.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the model to use
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self._client = None
        logging.basicConfig(level=logging.INFO)

    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def call(
        self, 
        prompt: str, 
        temperature: float = 0.5, 
        max_tokens: int = 1000, 
        response_format: Optional[Dict] = {"type": "json_object"}
    ) -> str:
        """
        Call the OpenAI model with retries and error handling.
        
        Args:
            prompt: The input prompt for the model
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens in the response
            response_format: Optional parameter to specify response format (default: JSON)
            
        Returns:
            The model's response content
        """
        try:
            # Prepare the base payload
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Add response_format unless explicitly set to None
            if response_format is not None:
                payload["response_format"] = response_format

            # Make the API call
            response = self.client.chat.completions.create(**payload)
            
            logger.debug(f"OpenAI Response: {response}")  # Log the raw response
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"OpenAI LLM call failed: {e}", exc_info=True)
            raise
    
    def _get_default_evaluation(self) -> str:
        """Return default response if processing fails."""
        return '{"error": "LLM evaluation failed"}'