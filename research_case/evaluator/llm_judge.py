import json
import logging
import os
from typing import Dict, List, Union
from openai import OpenAI

from research_case.analyzers.llm_client import LLMClient


logger = logging.getLogger(__name__)

class LLMJudge:
    """LLM-based judge for evaluating generated posts quality and authenticity."""
    
    def __init__(self,llm_client: LLMClient = LLMClient, model_name: str = "gpt-4o"):
        """
        Initialize LLM judge.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.llm_client = llm_client(api_key)
    
    def evaluate_post(self, 
                      original_post: str, 
                      generated_post: str, 
                      persona: Dict[str, str],
                      stimulus: str) -> Dict:
        """
        Evaluate a generated post using LLM judgment.
        
        Args:
            original_post: Original post text
            generated_post: Generated post text
            persona: Dictionary containing persona characteristics
            stimulus: The stimulus used for generation
            
        Returns:
            Dictionary containing evaluation scores and feedback
        """
        # Input validation
        if not isinstance(original_post, str) or not original_post.strip():
            logger.error("Invalid input: original_post must be a non-empty string.")
            return self._get_default_evaluation()

        if not isinstance(generated_post, str) or not generated_post.strip():
            logger.error("Invalid input: generated_post must be a non-empty string.")
            return self._get_default_evaluation()

        if not isinstance(persona, dict) or not persona:
            logger.error("Invalid input: persona must be a non-empty dictionary.")
            return self._get_default_evaluation()

        if not isinstance(stimulus, str) or not stimulus.strip():
            logger.error("Invalid input: stimulus must be a non-empty string.")
            return self._get_default_evaluation()

        prompt =  self._create_evaluation_prompt(
            original_post,
            generated_post,
            persona,
            stimulus
        )
        logger.info("Prompt successfully created for evaluation.")
        logger.debug(f"Prompt: {prompt}")

        try:
            response =  self.llm_client.call(
                prompt= prompt,
                temperature=0.1,
                max_tokens=500
            )
            logger.info("Received response from LLM.")
            
            logger.info(f'Raw Response: {response}')
            #response_content = json.loads(response)
            return self.parse_analysis(response)
            #return self.response_content
        except TypeError as te:
            logger.error(f"TypeError encountered: {te}")
            return self._get_default_evaluation()
        
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            
            return self._get_default_evaluation()


    def _create_evaluation_prompt(self,
                                  original_post: str,
                                  generated_post: str,
                                  persona: Dict[str, str],
                                  stimulus: str) -> str:
        """Create the evaluation prompt for the LLM."""
        return f"""Evaluate the generated social media post by comparing it to the original post based on the following criteria:

1. **Authenticity (1-10)**: How well does the generated post match the user’s persona? This includes tone, voice, and personal nuances.
2. **Style Consistency (1-10)**: How closely does the generated post maintain the style and structure of the original post?
3. **Matching Intent (Yes/No)**: Does the generated post align with the intent and message of the original post?


original social media post: {original_post}
generated social media post:{generated_post}


Provide the evaluation in the following JSON format:
{{
    "authenticity": {{"score": 1-10, "explanation": "brief explanation of the score"}},
    "style_consistency": {{"score": 1-10, "explanation": "brief explanation of the score"}},
    "matching_intent": true/false,
    "overall_feedback": "brief overall assessment of the generated post"
}}
"""

    def _get_default_evaluation(self) -> Dict:
        """Return default evaluation if LLM call fails."""
        return {
            "authenticity": {
                "score": 0,
                "explanation": "Evaluation failed"
            },
            "style_consistency": {
                "score": 0,
                "explanation": "Evaluation failed"
            },
            "matching_intent": False,
            "overall_feedback": "LLM evaluation failed"
        }

    @staticmethod
    def parse_analysis(response: str) -> Dict:
        """
        Parse the LLM response into a structured analysis dictionary.
        Handles boolean values and nested structures appropriately.
        """
        try:
            # Parse JSON response
            analysis = json.loads(response)
            
            # Required fields and their expected types
            required_fields = {
                "authenticity": dict,
                "style_consistency": dict,
                "matching_intent": bool,
                "overall_feedback": str
            }
            
            # Required subfields for dictionary fields
            required_subfields = {
                "authenticity": {"score": int, "explanation": str},
                "style_consistency": {"score": int, "explanation": str}
            }
            
            # Check required fields and their types
            for field, expected_type in required_fields.items():
                if field not in analysis:
                    raise KeyError(f"Missing required field: {field}")
                
                if not isinstance(analysis[field], expected_type):
                    raise TypeError(f"Field {field} must be of type {expected_type.__name__}")
                
                # Special handling for dictionary fields
                if expected_type == dict and field in required_subfields:
                    for subfield, subfield_type in required_subfields[field].items():
                        if subfield not in analysis[field]:
                            raise KeyError(f"Missing required subfield {subfield} in {field}")
                        
                        if not isinstance(analysis[field][subfield], subfield_type):
                            raise TypeError(f"Subfield {subfield} in {field} must be of type {subfield_type.__name__}")
                        
                        # Check for non-empty strings
                        if subfield_type == str and not str(analysis[field][subfield]).strip():
                            raise ValueError(f"Subfield {subfield} in {field} cannot be empty")
                
                # Check non-empty string for overall_feedback
                elif expected_type == str and not str(analysis[field]).strip():
                    raise ValueError(f"Field {field} cannot be empty")
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Invalid analysis structure: {e}")
            raise

    @staticmethod
    def load_json(file_path: str) -> Union[Dict, List]:
        """
        Load a JSON file and return its content.
        """
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            raise

    @staticmethod
    def save_json(data: Dict, file_path: str) -> None:
        """
        Save a dictionary as a JSON file.
        """
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save JSON file: {e}")
            raise
