"""Text similarity analysis module using jina-embeddings-v3."""

import os
import logging
from typing import Dict
from dataclasses import dataclass
import numpy as np
from transformers import AutoModel
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
# Define model constants


MODEL_NAME = "jinaai/jina-embeddings-v3"
MODEL_DIR = os.path.join('/Users/mogen/Desktop/Research_Case/embedding_model/', MODEL_NAME)

@dataclass
class SimilarityScores:
    """Container for similarity metrics."""
    semantic_similarity: float
    

class SimilarityAnalyzer:
    """Analyzer for computing similarity metrics between texts using jina-embeddings."""
    
    def __init__(self, max_length: int = 2048):
        """
        Initialize SimilarityAnalyzer with jina embeddings model.
        
        Args:
            max_length: Maximum input sequence length (default: 2048)
        """
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            self.model = model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

            self.max_length = max_length
            logger.info("Initialized jina-embeddings model successfully")
        except Exception as e:
            logger.error(f"Error initializing jina-embeddings model: {e}")
            raise
    
    def analyze_similarity(self, original: str, regenerated: str) -> SimilarityScores:
        """
        Analyze various similarity aspects between texts.
        
        Args:
            original: Original text
            regenerated: Regenerated text
            
        Returns:
            SimilarityScores object containing similarity metrics
        """
        try:
            semantic_similarity = self._compute_semantic_similarity(original, regenerated)
            
            
            return SimilarityScores(
                semantic_similarity=semantic_similarity,
                
            )
            
        except Exception as e:
            logger.error(f"Error analyzing similarity: {e}")
            return SimilarityScores(0.0, 0.0, 0.0)
    
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
            # Generate embeddings using text-matching task
            embeddings = self.model.encode(
                [text1, text2],
                task="text-matching",
                max_length=self.max_length
            )
            
            # Compute cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return 0.0