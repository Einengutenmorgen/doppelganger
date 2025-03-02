import json
import os
import logging
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import traceback2 as traceback
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import random
from random import sample
import re


from .llm_client import LLMClient
from .persona_prompt import PERSONA_FIELDS, PERSONA_ANALYSIS_PROMPT, EXAMPLE_PERSONA
from research_case.create_fields.persona_field_detector import EXAMPLE_FIELD_PERSONA, PERSONA_FIELDS_F, NEW_PERSONA_FIELD_ANALYSIS_PROMPT
from .prompt_generator import PersonaPromptGenerator

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class PersonaAnalysis:
    writing_style: str
    tone: str
    topics: List[str]
    personality_traits: List[str]
    engagement_patterns: Dict[str, str]
    language_preferences: Dict[str, str]

class PersonaAnalyzer:
    """
    Base PersonaAnalyzer class for persona analysis from file inputs.
    """
    def __init__(self, llm_client: LLMClient, 
                 max_retries: int = 3,
                 initial_wait: float = 1,
                 max_wait: float = 10):
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.max_wait = max_wait
        try:
            self.prompt_generator = PersonaPromptGenerator(
                base_prompt=PERSONA_ANALYSIS_PROMPT,
                example_persona=EXAMPLE_PERSONA,
                all_fields=PERSONA_FIELDS
            )
            logger.info("Initialized PersonaPromptGenerator")
        except Exception as e:
            logger.error(f"Failed to initialize PersonaPromptGenerator: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry attempt {retry_state.attempt_number} after error, waiting {retry_state.idle_for:.1f}s..."
        )
    )
    def _get_persona_with_retry(self, prompt: str) -> Dict:
        """Make LLM API call and parse response with unified retry logic"""
        try:
            # First make the API call
            response = self.llm_client.call(prompt)
            
            # Then try to parse it
            try:
                return self.parse_analysis(response)
            except (json.JSONDecodeError, KeyError, ValueError) as parse_error:
                logger.error(f"Parse failed: {str(parse_error)}, retrying entire operation")
                raise  # This will trigger a retry of both call and parse
                
        except Exception as e:
            logger.error(f"Operation failed: {str(e)}")
            raise  # This will trigger a retry of both call and parse

    def create_persona_prompt(self, posts: List[Dict], conversations: List[Dict], 
                            n_posts: int) -> str:

        """Base implementation for creating persona prompt"""
        posts_text = "\n".join(
            f"{i+1}. {post['full_text']}" for i, post in enumerate(posts[:n_posts])
        )

        conversations_text = ""
        if conversations and len(conversations) > 0:
            conversations_text = "\n".join(
                f"{i+1}. {conv['full_text']}" for i, conv in enumerate(conversations[:n_posts])
            )
            conversations_text = "\nUser Conversations:\n" + conversations_text
        return PERSONA_ANALYSIS_PROMPT.format(
            posts_text=posts_text,
            conversations_text=conversations_text,
            EXAMPLE_PERSONA=json.dumps(EXAMPLE_PERSONA, indent=4)
        )

    def analyze_persona_from_files(
        self, 
        posts_path: str, 
        conversations_path: Optional[str], 
        output_path: str, 
        n_posts: int = 5, 
        n_conversations: int = 5,
    ) -> None:
        """Base implementation of analyze_persona_from_files"""
        try:
            posts = self.load_json(posts_path)
            conversations = None
            if conversations_path and os.path.exists(conversations_path):
                conversations = self.load_json(conversations_path)

            persona_results = {}
            for user, user_posts in posts.items():
                user_conversations = []
                if conversations:
                    user_conversations = self.get_user_conversations(user, conversations, n_conversations)
                
                prompt = self.create_persona_prompt(user_posts, user_conversations, n_posts)
                response = self.llm_client.call(prompt)
                persona_results[user] = self.parse_analysis(response, PERSONA_FIELDS)
            
            self.save_json(persona_results, output_path)
            logger.info(f"Persona analysis complete. Results saved to {output_path}")

        except Exception as e:
            logger.error("Failed to analyze personas:")
            logger.error(traceback.format_exc())
            raise

    @staticmethod
    def parse_analysis(response: Union[str, Dict], selected_fields: List[str]) -> Dict:
        
        def convert_to_string(value) -> str:
            """Convert various data types to string format"""
            if isinstance(value, (list, tuple)):
                return ", ".join(str(item) for item in value)
            elif isinstance(value, dict):
                return "; ".join(f"{k}: {v}" for k, v in value.items())
            return str(value)
    
        try:
            # If response is a dictionary, use it directly
            if isinstance(response, dict):
                parsed_response = response
            elif isinstance(response, str):
                # Extract JSON using regex (removes any extra text before/after)
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    json_text = match.group(0)  # Extract JSON content
                    parsed_response = json.loads(json_text)
                else:
                    logger.error("Failed to extract valid JSON from response.")
                    return {field: "Error: No valid JSON detected" for field in selected_fields}
            else:
                logger.error("Unexpected response format.")
                return {field: "Error: Invalid response format" for field in selected_fields}
    
            # Process selected fields
            result = {}
            for field in selected_fields:
                if field not in parsed_response:
                    logger.warning(f"Missing required field: {field}")
                    result[field] = "N/A"
                    continue
    
                try:
                    result[field] = convert_to_string(parsed_response[field])
                    if not result[field].strip():
                        logger.warning(f"Field {field} is empty")
                        result[field] = "N/A"
                except Exception as e:
                    logger.error(f"Error converting field {field}: {e}")
                    result[field] = "N/A"
    
            return result
    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {field: "Error: Invalid JSON response" for field in selected_fields}
        except Exception as e:
            logger.error(f"Unexpected error during analysis parsing: {e}")
            return {field: "Error: Unexpected parsing error" for field in selected_fields}
    

    @staticmethod
    def load_json(file_path: str) -> Union[Dict, List]:
        """Load a JSON file and return its content."""
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_json(data: Dict, file_path: str) -> None:
        """Save a dictionary as a JSON file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
            if not os.path.exists(file_path):
                raise IOError(f"Failed to create file at {file_path}")
                
            file_size = os.path.getsize(file_path)
            logger.info(f"Successfully saved JSON file ({file_size} bytes) to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save JSON file to {file_path}: {str(e)}")
            raise

    @staticmethod
    def get_user_conversations(user: str, conversations: Union[Dict, List[Dict]], n_conversations: int) -> List[Dict]:
        """Extract top `n` conversations for a user from the conversation data."""
        if conversations is None:
            return []
            
        if isinstance(conversations, dict):
            user_convs = conversations.get(user, [])
            if not isinstance(user_convs, list):
                logger.error(f"Expected list of user conversations, got {type(user_convs)}")
                return []
                
        elif isinstance(conversations, list):
            user_convs = [
                conv for conv in conversations 
                if any(participant.get('id') == user for participant in conv.get('participants', []))
            ]
            
        else:
            logger.error(f"Expected dict or list of conversations, but got {type(conversations)}")
            return []

        user_convs.sort(key=lambda x: x.get("reply_count", 0), reverse=True)
        return user_convs[:n_conversations]


class PersonaFieldAnalyzer(PersonaAnalyzer):
    
    def create_num_fields_prompt(self, posts: List[Dict], conversations: List[Dict], 
                            n_posts: int) -> str:
        """Create prompt for field analysis"""
        posts_text = "\n".join(
            f"{i+1}. {post['full_text']}" for i, post in enumerate(posts[:n_posts])
        )
    
        conversations_text = ""
        if conversations and len(conversations) > 0:
            conversations_text = "\n".join(
                f"{i+1}. {conv['full_text']}" for i, conv in enumerate(conversations[:n_posts])
            )
            conversations_text = "\nUser Conversations:\n" + conversations_text
    
        formatted_prompt = NEW_PERSONA_FIELD_ANALYSIS_PROMPT.format(
            posts_text=posts_text,
            conversations_text=conversations_text,
            EXAMPLE_PERSONA=json.dumps(EXAMPLE_FIELD_PERSONA, indent=4)
        )
        
        return formatted_prompt

    def detect_the_fields(
        self, 
        posts_path: str, 
        conversations_path: Optional[str], 
        n_posts: int = 5, 
        n_conversations: int = 5
    ) -> List[str]:
        try:
            posts = self.load_json(posts_path)
            conversations = None
            if conversations_path and os.path.exists(conversations_path):
                conversations = self.load_json(conversations_path)
                    
            available_fields = []
            for user, user_posts in posts.items():
                user_conversations = []
                if conversations:
                    user_conversations = self.get_user_conversations(user, conversations, n_conversations)
            
            prompt = self.create_num_fields_prompt(
                user_posts, 
                user_conversations, 
                n_posts)
            
            response = self.llm_client.call(prompt)
            
            match = re.search(r'"present_categories": "([^"]*)"', response)
            if match:
                available_fields = match.group(1).split(", ")
            
            available_fields = list(set(available_fields))
            logger.info(f"Persona field analysis complete. Found fields: {available_fields}")
            return available_fields

        except Exception as e:
            logger.error("Failed to analyze persona fields:")
            logger.error(traceback.format_exc())
            raise
    

class ExtendedPersonaAnalyzer(PersonaAnalyzer):
    """
    Extended PersonaAnalyzer with additional prompt generation capabilities.
    """
    def create_persona_prompt(self, posts: List[Dict], conversations: List[Dict], 
                                n_posts: int,choosen_fields: List[str],  use_random_fields: bool = False, 
                            ) -> Union[str, Tuple[str, List[str]]]:
        """
        Extended implementation of create_persona_prompt with support for random fields.
        """
       
        if not use_random_fields:
            return super().create_persona_prompt(posts, conversations, n_posts)
        posts_text = "\n".join(
            f"{i+1}. {post['full_text']}" for i, post in enumerate(posts[:n_posts])
        )
        conversations_text = ""
        if conversations and len(conversations) > 0:
            conversations_text = "\n".join(
                f"{i+1}. {conv['full_text']}" for i, conv in enumerate(conversations[:n_posts])
            )
            conversations_text = "\nUser Conversations:\n" + conversations_text

        if not hasattr(self, '_prompt_generator'):
            self._prompt_generator = PersonaPromptGenerator(
                base_prompt="",  # Base prompt is built into the generator
                example_persona=EXAMPLE_PERSONA,
                all_fields=PERSONA_FIELDS
            )
        # bu kısımda selected fields artık önemli değil 
        prompt_template, choosen_fields = self._prompt_generator.generate_prompt_version(choosen_fields)
        
        formatted_prompt = prompt_template.format(
            posts_text=posts_text,
            conversations_text=conversations_text
        )
        
        return formatted_prompt, choosen_fields
        
    def analyze_persona_from_files(
        self, 
        posts_path: str, 
        conversations_path: Optional[str], 
        founded_fields: List[str],
        output_path: str, 
        n_posts: int = 5, 
        n_conversations: int = 5
    ) -> None:
        """
        Extended version of analyze_persona_from_files that handles random fields.
        """
        try:
            posts = self.load_json(posts_path)
            conversations = None
            if conversations_path and os.path.exists(conversations_path):
                conversations = self.load_json(conversations_path)
                    
            persona_results = {}
            for user, user_posts in posts.items():
                user_conversations = []
                if conversations:
                    user_conversations = self.get_user_conversations(user, conversations, n_conversations)
                    
            number_of_random_fields = random.randint(1, len(founded_fields))
            randomly_choosen_fields = random.sample(founded_fields, number_of_random_fields)
            
            prompt = self.create_persona_prompt(
                    user_posts,
                    user_conversations, 
                    n_posts,
                    choosen_fields=randomly_choosen_fields,
                    use_random_fields=True
                )

            if isinstance(prompt, tuple):
                prompt = prompt[0]
                
            response = self.llm_client.call(prompt)
            parsed_result = self.parse_analysis(response, randomly_choosen_fields)
            #self.save_json(persona_results, output_path)

            # Load existing results if the file exists
            existing_results = self.load_json(output_path) if os.path.exists(output_path) else {}
            if user not in existing_results:
                existing_results[user] = []
            
            existing_results[user].append(parsed_result)
            
            self.save_json(existing_results, output_path)
            logger.info(f"Persona analysis complete. Results saved to {output_path}")

        except Exception as e:
            logger.error("Failed to analyze personas:")
            logger.error(traceback.format_exc())
            raise