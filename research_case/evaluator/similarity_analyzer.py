"""Text similarity analysis module using jina-embeddings-v3."""
import os
import logging
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Define model constants
MODEL_NAME = "jinaai/jina-embeddings-v3"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'embedding_model', MODEL_NAME)


class SimilarityAnalyzer:
    """Analyzer for computing similarity metrics between texts using jina-embeddings."""
    
    def __init__(self, max_length: int = 2048):
        """
        Initialize SimilarityAnalyzer with jina embeddings model.
        
        Args:
            max_length: Maximum input sequence length (default: 2048)
        """
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        
    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            try:
                os.makedirs(MODEL_DIR, exist_ok=True)
                self._model = AutoModel.from_pretrained(
                    "jinaai/jina-embeddings-v3", 
                    trust_remote_code=True
                )
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "jinaai/jina-embeddings-v3"
                )
                logger.info("Initialized jina-embeddings model successfully")
            except Exception as e:
                logger.error(f"Error initializing jina-embeddings model: {e}")
                raise
    
    @property
    def model(self):
        """Get the model, loading it if necessary."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Get the tokenizer, loading it if necessary."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def analyze_similarity(self, original: str, regenerated: str) -> float:
        """
        Analyze various similarity aspects between texts.
        
        Args:
            original: Original text
            regenerated: Regenerated text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            semantic_similarity = self._compute_semantic_similarity(original, regenerated)
            return semantic_similarity
        except Exception as e:
            logger.error(f"Error analyzing similarity: {e}")
            raise  # Better to raise the error than return 0.0

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts using jina embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Semantic similarity score
        """
        try:
            import torch
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Ensure model is loaded
            self._load_model()
            
            # Tokenize and encode texts
            inputs = self.tokenizer(
                [text1, text2],
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model(**inputs).last_hidden_state[:, 0, :]  # Using [CLS] token
            
            # Compute cosine similarity
            similarity: float = cosine_similarity(
                embeddings[0].reshape(1, -1), 
                embeddings[1].reshape(1, -1)
            )[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            raise