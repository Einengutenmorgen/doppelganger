"""Main evaluation pipeline module."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from .rouge_evaluator import RougeEvaluator
from .similarity_analyzer import SimilarityAnalyzer
from .llm_judge import LLMJudge

logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """Pipeline for evaluating generated social media posts."""
    
    def __init__(self):
        """Initialize evaluation components."""
        self.rouge_evaluator = RougeEvaluator()
        self.similarity_analyzer = SimilarityAnalyzer()
        self.llm_judge = LLMJudge()
        
    def evaluate_post(self, original_post: Dict, generated_post: Dict, persona: Dict) -> Dict:
        """
        Evaluate a single generated post.
        
        Args:
            original_post: Original post data
            generated_post: Generated post data
            persona: User persona data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Basic metrics
            rouge_scores = self.rouge_evaluator.calculate_scores(
                original_post['full_text'],
                generated_post['generated_text']
            )
            
            similarity_scores = self.similarity_analyzer.analyze_similarity(
                original=original_post['full_text'],
                regenerated=generated_post['generated_text'],
                neutral=generated_post['stimulus']
            )
            
            # LLM evaluation
            llm_evaluation = self.llm_judge.evaluate_post(
                original_post=original_post['full_text'],
                generated_post=generated_post['generated_text'],
                persona=persona,
                stimulus=generated_post['stimulus']
            )
            
            return {
                'rouge_scores': rouge_scores,
                'similarity_scores': similarity_scores.__dict__,
                'llm_evaluation': llm_evaluation,
                'metadata': {
                    'original_id': original_post.get('tweet_id'),
                    'generated_id': generated_post.get('generation_id'),
                    'timestamp': generated_post.get('generation_timestamp')
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
        evaluations = []
        for post in data['generated_posts']:
            try:
                evaluation = self.evaluate_post(
                    original_post={'tweet_id': post['original_post_id'], 'full_text': post['original_text']},
                    generated_post=post,
                    persona={
                        'writing_style': post['persona_writing_style'],
                        'tone': post['persona_tone'],
                        'topics': post['persona_topics']
                    }
                )
                evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Error evaluating post {post.get('generation_id')}: {e}")
                
        return {
            'individual_evaluations': evaluations,
            'aggregate_metrics': self._calculate_aggregate_metrics(evaluations),
            'metadata': {
                'total_evaluated': len(evaluations),
                'evaluation_timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
    
    def _calculate_aggregate_metrics(self, evaluations: List[Dict]) -> Dict:
        """Calculate aggregate statistics for all metrics."""
        try:
            rouge_scores = [e['rouge_scores'] for e in evaluations]
            similarity_scores = [e['similarity_scores'] for e in evaluations]
            llm_scores = [e['llm_evaluation'] for e in evaluations]
            
            return {
                'rouge': self._aggregate_rouge_scores(rouge_scores),
                'similarity': self._aggregate_similarity_scores(similarity_scores),
                'llm_evaluation': self._aggregate_llm_scores(llm_scores)
            }
        except Exception as e:
            logger.error(f"Error calculating aggregate metrics: {e}")
            return {}
    
    def _get_default_evaluation(self) -> Dict:
        """Return default evaluation if processing fails."""
        return {
            'rouge_scores': {},
            'similarity_scores': {},
            'llm_evaluation': self.llm_judge._get_default_evaluation(),
            'metadata': {
                'error': 'Evaluation failed',
                'timestamp': datetime.now().isoformat()
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
    def _aggregate_similarity_scores(scores: List[Dict]) -> Dict:
        """Aggregate similarity scores across multiple evaluations."""
        if not scores:
            return {}
            
        return {
            'semantic_similarity': {
                'mean': sum(s['semantic_similarity'] for s in scores) / len(scores),
                'std': 0.0
            },
            'style_similarity': {
                'mean': sum(s['style_similarity'] for s in scores) / len(scores),
                'std': 0.0
            },
            'content_preservation': {
                'mean': sum(s['content_preservation'] for s in scores) / len(scores),
                'std': 0.0
            }
        }
    
    @staticmethod
    def _aggregate_llm_scores(scores: List[Dict]) -> Dict:
        """Aggregate LLM evaluation scores across multiple evaluations."""
        if not scores:
            return {}
            
        return {
            'authenticity': {
                'mean': sum(s['authenticity']['score'] for s in scores) / len(scores),
                'std': 0.0
            },
            'response_quality': {
                'mean': sum(s['response_quality']['score'] for s in scores) / len(scores),
                'std': 0.0
            },
            'style_consistency': {
                'mean': sum(s['style_consistency']['score'] for s in scores) / len(scores),
                'std': 0.0
            }
        }