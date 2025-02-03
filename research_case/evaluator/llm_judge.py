import json
import logging
import os
from typing import Dict, List, Union
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from research_case.analyzers.llm_client import LLMClient
from research_case.LLMclients.llm_client_google import GeminiLLMClient


logger = logging.getLogger(__name__)

    
class LLMJudge:
    """LLM-based judge for evaluating generated posts quality and authenticity."""
    
    def __init__(self, 
                client_type: str = "gemini",
                model_name: str = None):
        """
        Initialize LLM judge with specified client type.
        
        Args:
            client_type: Type of LLM client to use ("openai" or "gemini")
            model_name: Name of the model to use (optional)
        """
        # Get appropriate API key based on client type
        if client_type == "openai":
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            self.llm_client = OpenAI(api_key=api_key)
            self.model_name = model_name or "gpt-4"
            
        elif client_type == "gemini":
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set.")
            self.llm_client = GeminiLLMClient(api_key=api_key)
            # Note: model_name is handled internally for Gemini client
            
        else:
            raise ValueError(f"Unsupported client_type: {client_type}. Use 'openai' or 'gemini'.")
        
        self.client_type = client_type
        logger.info(f"Initialized LLMJudge with {client_type} client")
    
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

        prompt = self._create_evaluation_prompt(
            original_post,
            generated_post,
            persona,
            stimulus
        )
        logger.info("Prompt successfully created for evaluation.")
        logger.debug(f"Prompt: {prompt}")

        try:
            # Handle different client types
            if self.client_type == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                response_text = response.choices[0].message.content
            else:  # gemini
                response_text = self.llm_client.call(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=500
                )
            
            logger.info("Received response from LLM.")
            logger.debug(f'Raw Response: {response_text}')
            
            return self.parse_analysis(response_text)
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return self._get_default_evaluation()

    def _create_evaluation_prompt(self,
                                original_post: str,
                                generated_post: str,
                                persona: Dict[str, str],
                                stimulus: str) -> str:
        """Create the evaluation prompt for the LLM."""
        return f"""You are an expert evaluator assessing the quality of an AI-generated social media post. Your task is to compare the generated post with the original and provide a structured assessment based on the following criteria:

1. **Authenticity (1-10):**  
   - Does the generated post reflect the unique tone, voice, and personality of the original author?  
   - Does it maintain their typical word choices, phrasing, and emotional nuances?  
   - Justify the score with a brief explanation.  

2. **Style Consistency (1-10):**  
   - How well does the generated post match the writing style of the original?  
   - Does it retain the sentence structure, rhythm, and distinctive linguistic patterns (idiolect/sociolect)?  
   - Provide an explanation for the score.  

3. **Intent Matching (True/False):**  
   - Does the generated post preserve the key message, emotional impact, and overall intent of the original post?  
   - If False, briefly explain the discrepancies.  

**Original Post:**  
{original_post}

**Generated Post:**  
{generated_post}

**Persona Information:**
{json.dumps(persona, indent=2)}

**Stimulus:**
{stimulus}

Return the evaluation in the following structured JSON format:
{{
    "authenticity": {{
        "score": <integer 1-10>,
        "explanation": "<brief explanation>"
    }},
    "style_consistency": {{
        "score": <integer 1-10>,
        "explanation": "<brief explanation>"
    }},
    "matching_intent": <true/false>,
    "overall_feedback": "<brief assessment summarizing the strengths and weaknesses of the generated post>"
}}"""

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
