
import os
import json
import logging
from openai import OpenAI
from typing import Dict, List, Optional
from research_case.analyzers.llm_client import LLMClient

logger = logging.getLogger(__name__)

class ConversationJudge:
    """LLM-based judge for evaluating conversation responses in context."""
    
    def __init__(self, llm_client: LLMClient = None):
        """
        Initialize conversation judge.
        
        Args:
            llm_client: LLMClient instance (optional)
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        if llm_client is None:
            self.llm_client = LLMClient(api_key=api_key)
        else:
            self.llm_client = llm_client
        
        self.model_name = 'gpt-4o'

    def evaluate_response(self,
                         original_response: str,
                         generated_response: str,
                         conversation_history: List[Dict],
                         persona: Dict,
                         parent_message: str) -> Dict:
        """
        Evaluate a generated response in conversation context.
        
        Args:
            original_response: Original response text
            generated_response: Generated response text
            conversation_history: List of prior messages in conversation
            persona: Dictionary containing persona characteristics
            parent_message: The message being replied to
            
        Returns:
            Dictionary containing evaluation scores and feedback
        """
        try:
            prompt = self._create_evaluation_prompt(
                original_response=original_response,
                generated_response=generated_response,
                conversation_history=conversation_history,
                persona=persona,
                parent_message=parent_message
            )
            
            response = self.llm_client.call(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500
            )
            
            return self.parse_evaluation(response)
            
        except Exception as e:
            logger.error(f"Error in conversation evaluation: {e}")
            return self._get_default_evaluation()

    def _create_evaluation_prompt(self,
                                original_response: str,
                                generated_response: str,
                                conversation_history: List[Dict],
                                persona: Dict,
                                parent_message: str) -> str:
        """Create the evaluation prompt including conversation context."""
        
        # Format conversation history
        history_text = self._format_conversation_history(conversation_history)
        
        return f"""Evaluate this generated social media response in the context of its conversation thread and the user's persona.

CONVERSATION CONTEXT:
Previous messages in thread:
{history_text}

Parent message being replied to:
"{parent_message}"

RESPONSES TO EVALUATE:
Original Response: "{original_response}"
Generated Response: "{generated_response}"

Please evaluate the generated response on these criteria:

1. Response Appropriateness (1-10):
- How well does it address the parent message?
- Is it a natural and fitting response in the conversation?
- Does it maintain appropriate tone and context?

2. Persona Consistency (1-10):
- Does it match the persona's writing style?
- Does it maintain the persona's typical tone?
- Is it consistent with the persona's interests/topics?

3. Context Awareness (1-10):
- Does it show understanding of the conversation history?
- Does it maintain thread continuity?
- Does it reference relevant context appropriately?

4. Natural Flow (Yes/No):
- Does it feel like a natural part of the conversation?
- Would it make sense to other participants?

Provide the evaluation in the following JSON format:
{{
    "response_appropriateness": {{
        "score": 1-10,
        "explanation": "brief explanation of score"
    }},
    "persona_consistency": {{
        "score": 1-10,
        "explanation": "brief explanation of score"
    }},
    "context_awareness": {{
        "score": 1-10,
        "explanation": "brief explanation of score"
    }},
    "natural_flow": {{
        "value": true/false,
        "explanation": "brief explanation of assessment"
    }},
    "overall_feedback": "brief overall assessment of the response"
}}"""

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history for the prompt."""
        formatted = []
        for msg in history:
            formatted.append(
                f"@{msg.get('screen_name', 'user')}: {msg.get('full_text', '')}"
            )
        return "\n".join(formatted)

    def parse_evaluation(self, response: str) -> Dict:
        """Parse the LLM response into a structured evaluation dictionary."""
        try:
            evaluation = json.loads(response)
            
            # Validate required fields
            required_fields = {
                'response_appropriateness': dict,
                'persona_consistency': dict,
                'context_awareness': dict,
                'natural_flow': dict,
                'overall_feedback': str
            }
            
            for field, expected_type in required_fields.items():
                if field not in evaluation:
                    raise KeyError(f"Missing required field: {field}")
                if not isinstance(evaluation[field], expected_type):
                    raise TypeError(f"Invalid type for {field}")
                    
            # Validate score fields
            score_fields = ['response_appropriateness', 'persona_consistency', 'context_awareness']
            for field in score_fields:
                score = evaluation[field].get('score')
                if not isinstance(score, (int, float)) or score < 1 or score > 10:
                    raise ValueError(f"Invalid score in {field}")
                    
            # Validate natural flow
            flow = evaluation['natural_flow'].get('value')
            if not isinstance(flow, bool):
                raise ValueError("Invalid natural_flow value")
                
            return evaluation
            
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error(f"Error parsing evaluation: {e}")
            return self._get_default_evaluation()

    def _get_default_evaluation(self) -> Dict:
        """Return default evaluation if processing fails."""
        return {
            "response_appropriateness": {
                "score": 0,
                "explanation": "Evaluation failed"
            },
            "persona_consistency": {
                "score": 0,
                "explanation": "Evaluation failed"
            },
            "context_awareness": {
                "score": 0,
                "explanation": "Evaluation failed"
            },
            "natural_flow": {
                "value": False,
                "explanation": "Evaluation failed"
            },
            "overall_feedback": "Evaluation failed"
        }