"""ROUGE metrics evaluation module."""

import logging
from typing import Dict, List
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

class RougeEvaluator:
    """Calculator for ROUGE metrics between original and generated posts."""
    
    def __init__(self):
        """Initialize RougeEvaluator with default metrics."""
        self.metrics = ['rouge1', 'rouge2', 'rougeL']
        self.scorer = rouge_scorer.RougeScorer(self.metrics, use_stemmer=True)
    
    def calculate_scores(self, original_text: str, generated_text: str) -> Dict:
        """
        Calculate ROUGE scores between original and generated text.
        
        Args:
            original_text: Original text
            generated_text: Generated/regenerated text
            
        Returns:
            Dictionary containing ROUGE scores for each metric
        """
        try:
            scores = self.scorer.score(original_text, generated_text)
            
            return {
                metric: {
                    'precision': scores[metric].precision,
                    'recall': scores[metric].recall,
                    'fmeasure': scores[metric].fmeasure
                }
                for metric in self.metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {
                metric: {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0} 
                for metric in self.metrics
            }