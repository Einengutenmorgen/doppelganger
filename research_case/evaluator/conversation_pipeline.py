"""Conversation evaluation pipeline module."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from research_case.evaluator.rouge_evaluator import RougeEvaluator
from research_case.evaluator.similarity_analyzer import SimilarityAnalyzer
from research_case.evaluator.conversation_judge import ConversationJudge

logger = logging.getLogger(__name__)

@dataclass
class ConversationEvaluation:
    """Container for conversation evaluation inputs"""
    original_response: str
    generated_response: str
    conversation_history: List[Dict]
    parent_message: str
    persona: Dict
    metadata: Optional[Dict] = None

class ConversationEvaluationPipeline:
    """Pipeline for evaluating generated conversation responses."""
    
    def __init__(self):
        """Initialize evaluation components."""
        self.rouge_evaluator = RougeEvaluator()
        self.similarity_analyzer = SimilarityAnalyzer()
        self.conversation_judge = ConversationJudge()
    
    def evaluate_response(self, evaluation: ConversationEvaluation) -> Dict:
        """
        Evaluate a single generated response.
        
        Args:
            evaluation: ConversationEvaluation containing response and context
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Text similarity metrics
            rouge_scores = self.rouge_evaluator.calculate_scores(
                original_text=evaluation.original_response,
                generated_text=evaluation.generated_response
            )
            
            semantic_similarity = self.similarity_analyzer.analyze_similarity(
                original=evaluation.original_response,
                regenerated=evaluation.generated_response
            )
            
            # Conversation-specific LLM evaluation
            llm_evaluation = self.conversation_judge.evaluate_response(
                original_response=evaluation.original_response,
                generated_response=evaluation.generated_response,
                conversation_history=evaluation.conversation_history,
                persona=evaluation.persona,
                parent_message=evaluation.parent_message
            )
            
            return {
                'rouge_scores': rouge_scores,
                'semantic_similarity': semantic_similarity,
                'conversation_metrics': llm_evaluation,
                'metadata': evaluation.metadata or {},
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return self._get_default_evaluation()
    
    def evaluate_batch(self, generated_data: Dict) -> Dict:
        """
        Evaluate all generated responses in the dataset.
        
        Args:
            generated_data: Dictionary containing generated responses data
            
        Returns:
            Dictionary containing evaluation results and statistics
        """
        evaluations = []
        
        for response in generated_data['generated_responses']:
            try:
                conversation_id = response.get('conversation_id')
                
                evaluation = ConversationEvaluation(
                    original_response=response.get('original_text', ''),
                    generated_response=response.get('generated_text', ''),
                    conversation_history=self._get_conversation_history(
                        conversation_id, 
                        generated_data
                    ),
                    parent_message=response.get('parent_message', ''),
                    persona={
                        'writing_style': response.get('persona_writing_style', ''),
                        'tone': response.get('persona_tone', ''),
                        'topics': response.get('persona_topics', '')
                    },
                    metadata={
                        'conversation_id': conversation_id,
                        'generation_id': response.get('generation_id'),
                        'original_timestamp': response.get('original_timestamp'),
                        'generation_timestamp': response.get('generation_timestamp')
                    }
                )
                
                result = self.evaluate_response(evaluation)
                evaluations.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating response {response.get('generation_id')}: {e}")
        
        return {
            'individual_evaluations': evaluations,
            'aggregate_metrics': self._calculate_aggregate_metrics(evaluations),
            'metadata': {
                'total_evaluated': len(evaluations),
                'evaluation_timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
    
    def _get_conversation_history(self, conversation_id: str, data: Dict) -> List[Dict]:
        """
        Extract conversation history for a specific conversation.
        
        Args:
            conversation_id: ID of the conversation
            data: Dictionary containing all conversations data
            
        Returns:
            List of messages in chronological order
        """
        try:
            # Get raw conversation if it exists in the input data
            raw_conversation = data.get('conversation_data', {}).get(conversation_id, [])
            
            if not raw_conversation:
                logger.warning(f"No conversation found for ID: {conversation_id}")
                return []
            
            # Sort messages by timestamp to ensure chronological order
            sorted_messages = sorted(
                raw_conversation,
                key=lambda x: x.get('created_at', ''),
                reverse=False  # Oldest first
            )
            
            # Format each message in the conversation
            formatted_messages = []
            for msg in sorted_messages:
                formatted_messages.append({
                    'user_id': msg.get('original_user_id'),
                    'screen_name': msg.get('screen_name'),
                    'text': msg.get('full_text'),
                    'timestamp': msg.get('created_at'),
                    'tweet_id': msg.get('tweet_id'),
                    'reply_to_id': msg.get('reply_to_id')
                })
                
            return formatted_messages
            
        except Exception as e:
            logger.error(f"Error extracting conversation history: {e}")
            return []

    def _calculate_aggregate_metrics(self, evaluations: List[Dict]) -> Dict:
        """Calculate aggregate statistics across all evaluations."""
        try:
            valid_evals = [e for e in evaluations if self._is_valid_evaluation(e)]
            
            if not valid_evals:
                logger.warning("No valid evaluations to aggregate")
                return {}

            return {
                "rouge_metrics": self._aggregate_rouge_scores(valid_evals),
                "similarity_metrics": self._aggregate_similarity_scores(valid_evals),
                "conversation_metrics": self._aggregate_conversation_metrics(valid_evals)
            }
        except Exception as e:
            logger.error(f"Error calculating aggregate metrics: {e}")
            return {}

    def _get_default_evaluation(self) -> Dict:
        """Return default evaluation if processing fails."""
        return {
            'rouge_scores': {},
            'semantic_similarity': 0.0,
            'conversation_metrics': self.conversation_judge._get_default_evaluation(),
            'metadata': {
                'error': 'Evaluation failed',
                'timestamp': datetime.now().isoformat()
            }
        }

    @staticmethod
    def _is_valid_evaluation(evaluation: Dict) -> bool:
        """Check if an evaluation result is valid and complete."""
        required_keys = ['rouge_scores', 'semantic_similarity', 'conversation_metrics']
        return all(key in evaluation for key in required_keys)

    @staticmethod
    def _aggregate_rouge_scores(evaluations: List[Dict]) -> Dict:
        """Aggregate ROUGE scores across multiple evaluations."""
        scores = {}
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            scores[metric] = {
                'precision': {
                    'mean': sum(e['rouge_scores'][metric]['precision'] 
                              for e in evaluations) / len(evaluations)
                },
                'recall': {
                    'mean': sum(e['rouge_scores'][metric]['recall'] 
                              for e in evaluations) / len(evaluations)
                },
                'fmeasure': {
                    'mean': sum(e['rouge_scores'][metric]['fmeasure'] 
                              for e in evaluations) / len(evaluations)
                }
            }
        return scores

    @staticmethod
    def _aggregate_similarity_scores(evaluations: List[Dict]) -> Dict:
        """Aggregate similarity scores across multiple evaluations."""
        similarities = [e['semantic_similarity'] for e in evaluations]
        return {
            'mean': sum(similarities) / len(similarities)
        }

    @staticmethod
    def _aggregate_conversation_metrics(evaluations: List[Dict]) -> Dict:
        """Aggregate conversation-specific metrics across evaluations."""
        metrics = {}
        for field in ['response_appropriateness', 'persona_consistency', 'context_awareness']:
            scores = [e['conversation_metrics'][field]['score'] for e in evaluations]
            metrics[field] = {
                'mean': sum(scores) / len(scores)
            }
            
        # Calculate natural flow ratio
        flow_count = sum(1 for e in evaluations 
                        if e['conversation_metrics']['natural_flow']['value'])
        metrics['natural_flow_ratio'] = flow_count / len(evaluations)
        
        return metrics

    def save_results(self, results: Dict, output_path: str) -> None:
        """Save evaluation results to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")