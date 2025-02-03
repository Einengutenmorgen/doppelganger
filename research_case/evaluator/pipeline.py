"""Main evaluation pipeline module."""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .rouge_evaluator import RougeEvaluator
from .similarity_analyzer import SimilarityAnalyzer
from .llm_judge import LLMJudge

logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """Pipeline for evaluating generated social media posts."""
    
    def __init__(self, llm_client_type: str = "gemini"):
        """
        Initialize evaluation components.
        
        Args:
            llm_client_type: Type of LLM client to use ("openai" or "gemini")
        """
        self._rouge_evaluator = None
        self._similarity_analyzer = None
        self._llm_judge = LLMJudge(client_type=llm_client_type)
        logger.info(f"Initialized EvaluationPipeline with {llm_client_type} LLM client")
    
    @property
    def rouge_evaluator(self):
        """Lazy load RougeEvaluator."""
        if self._rouge_evaluator is None:
            self._rouge_evaluator = RougeEvaluator()
        return self._rouge_evaluator
    
    @property
    def similarity_analyzer(self):
        """Lazy load SimilarityAnalyzer."""
        if self._similarity_analyzer is None:
            self._similarity_analyzer = SimilarityAnalyzer()
        return self._similarity_analyzer
    
    def evaluate_post(self, post_data: Dict) -> Dict:
        """
        Evaluate a single generated post.
        
        Args:
            post_data: Dictionary containing post data with the following structure:
                {
                    "original_text": str,
                    "generated_text": str,
                    "persona": dict,
                    "stimulus": str,
                    ...
                }
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Extract required fields
            original_text = post_data["original_text"]
            generated_text = post_data["generated_text"]
            
            # Extract persona fields (all fields starting with "persona_")
            persona = {
                k.replace("persona_", ""): v
                for k, v in post_data.items()
                if k.startswith("persona_")
            }
            
            # Get stimulus
            stimulus = post_data.get("stimulus", "")
            
            # Basic metrics
            rouge_scores = self.rouge_evaluator.calculate_scores(
                original_text,
                generated_text
            )
            
            similarity_scores = self.similarity_analyzer.analyze_similarity(
                original=original_text,
                regenerated=generated_text,
            )
            
            # LLM evaluation
            llm_evaluation = self._llm_judge.evaluate_post(
                original_post=original_text,
                generated_post=generated_text,
                persona=persona,
                stimulus=stimulus
            )
            
            return {
                'rouge_scores': rouge_scores,
                'similarity_scores': similarity_scores,
                'llm_evaluation': llm_evaluation,
                'metadata': {
                    'original_id': post_data.get('original_post_id'),
                    'generated_id': post_data.get('generation_id'),
                    'timestamp': post_data.get('generation_timestamp')
                }
            }
            
        except Exception as e:
            logger.error(f"Error evaluating post: {e}")
            return self._get_default_evaluation()
    
    def evaluate_batch(self, data: Dict) -> Dict:
        """
        Evaluate all generated posts in the dataset.
        
        Args:
            data: Dictionary containing generated posts data
            
        Returns:
            Dictionary containing evaluation results and statistics
        """
        # Handle both list and dict input formats
        if isinstance(data, dict):
            posts = data.get('generated_posts', [])
        elif isinstance(data, list):
            posts = data
        else:
            logger.error(f"Invalid input data type: {type(data)}")
            return self._get_default_evaluation()
        
        evaluations = []
        for post in posts:
            try:
                evaluation = self.evaluate_post(post)
                evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Error evaluating post {post.get('generation_id')}: {e}")
                
        return {
            'individual_evaluations': evaluations,
            'aggregate_metrics': self._calculate_aggregate_metrics(evaluations),
            'metadata': {
                'total_evaluated': len(evaluations),
                'evaluation_timestamp': datetime.now(timezone.utc).isoformat(),
                'version': '1.0'
            }
        }
    
    def _calculate_aggregate_metrics(self, evaluations: List[Dict]) -> Dict:
        """Calculate aggregate statistics for all metrics."""
        try:
            valid_evaluations = [e for e in evaluations if e and all(k in e for k in ["rouge_scores", "llm_evaluation", "similarity_scores"])]
            
            if not valid_evaluations:
                logger.warning("No valid evaluations to aggregate")
                return {}

            return {
                "rouge": self._aggregate_rouge_scores([e["rouge_scores"] for e in valid_evaluations]),
                "llm_evaluation": self._aggregate_llm_scores([e["llm_evaluation"] for e in valid_evaluations]),
                "similarity_scores": self._aggregate_similarity_scores([e["similarity_scores"] for e in valid_evaluations])
            }
        except Exception as e:
            logger.error(f"Error calculating aggregate metrics: {e}")
            return {}
    
    def _get_default_evaluation(self) -> Dict:
        """Return default evaluation if processing fails."""
        return {
            'rouge_scores': {},
            'similarity_scores': {},
            'llm_evaluation': self._llm_judge._get_default_evaluation(),
            'metadata': {
                'error': 'Evaluation failed',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    @staticmethod
    def _aggregate_rouge_scores(scores: List[Dict]) -> Dict:
        """Aggregate ROUGE scores across multiple evaluations."""
        if not scores:
            return {}
            
        metrics = scores[0].keys()
        aggregated = {}
        
        for metric in metrics:
            aggregated[metric] = {
                'precision': {
                    'mean': sum(s[metric]['precision'] for s in scores) / len(scores),
                    'std': 0.0  # Calculate std if needed
                },
                'recall': {
                    'mean': sum(s[metric]['recall'] for s in scores) / len(scores),
                    'std': 0.0
                },
                'fmeasure': {
                    'mean': sum(s[metric]['fmeasure'] for s in scores) / len(scores),
                    'std': 0.0
                }
            }
        
        return aggregated
    
    @staticmethod
    def _aggregate_similarity_scores(scores: List[float]) -> Dict:
        """Aggregate similarity scores across multiple evaluations."""
        if not scores:
            return {}
            
        return {
            'mean': sum(scores) / len(scores),
            'std': 0.0  # Calculate std if needed
        }
    
    @staticmethod
    def _aggregate_llm_scores(scores: List[Dict]) -> Dict:
        """Aggregate LLM evaluation scores across multiple evaluations."""
        if not scores:
            return {}
                
        try:
            aggregated = {
                "authenticity": {
                    "mean": sum(s["authenticity"]["score"] for s in scores) / len(scores),
                    "std": 0.0  # Calculate std if needed
                },
                "style_consistency": {
                    "mean": sum(s["style_consistency"]["score"] for s in scores) / len(scores),
                    "std": 0.0
                }
            }

            # Count matching_intent true/false ratio
            matching_intent_count = sum(1 for s in scores if s.get("matching_intent", False))
            aggregated["matching_intent_ratio"] = matching_intent_count / len(scores)

            return aggregated
        except Exception as e:
            logger.error(f"Error aggregating LLM scores: {e}")
            return {}