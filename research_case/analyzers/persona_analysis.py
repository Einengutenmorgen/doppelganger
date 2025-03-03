import json
import os
import logging
import re
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import traceback2 as traceback
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .llm_client import LLMClient
from .persona_prompt import PERSONA_FIELDS, PERSONA_ANALYSIS_PROMPT, EXAMPLE_PERSONA
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

    def load_and_sample_users(input_path: str, max_personas: int, min_posts_per_user: int) -> dict:
        """
        Load user data from JSON and sample up to max_personas users, filtering those with minimum required posts.
        
        Args:
            input_path: Path to input JSON file
            max_personas: Maximum number of personas to create
            min_posts_per_user: Minimum number of posts required per user (default: 0)
            
        Returns:
            Dict containing sampled user data
        """
        try:
            with open(input_path, 'r') as f:
                all_users = json.load(f)
                
            # Filter users with minimum required posts
            if min_posts_per_user > 0:
                filtered_users = {uid: posts for uid, posts in all_users.items() if len(posts) >= min_posts_per_user}
                logger.info(f"Filtered from {len(all_users)} to {len(filtered_users)} users with at least {min_posts_per_user} posts")
                all_users = filtered_users
                
            # If max_personas is specified and less than total users, sample randomly
            if max_personas and len(all_users) > max_personas:
                import random
                sampled_user_ids = random.sample(list(all_users.keys()), max_personas)
                sampled_users = {uid: all_users[uid] for uid in sampled_user_ids}
                logger.info(f"Sampled {max_personas} users from total {len(all_users)} users")
                return sampled_users
                
            return all_users
        except Exception as e:
            logger.error(f"Error loading user data: {e}")
            raise

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
        n_conversations: int = 5
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
    def fix_json_string(json_str):
        """
        Attempt to fix common JSON string issues:
        1. Unterminated strings
        2. Unescaped quotes within strings
        3. Trailing commas
        4. Missing closing brackets
        
        Returns the fixed JSON string or the original if fixing fails.
        """
        try:
            # First try to parse as is
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            logger.info(f"Attempting to fix JSON error: {e}")
            
            fixed_str = json_str
            
            # Fix unterminated strings - look for keys without closing quotes
            key_pattern = r'("[\w_]+)(\s*:)'
            fixed_str = re.sub(key_pattern, r'\1"\2', fixed_str)
            
            # Fix unescaped quotes in strings
            # This is a simplified approach - a more robust solution would be more complex
            value_pattern = r':(\s*)"([^"]*?)([^\\])"([^,}\]])'
            fixed_str = re.sub(value_pattern, r':\1"\2\3\\"\4', fixed_str)
            
            # Fix trailing commas in objects
            fixed_str = re.sub(r',(\s*})', r'\1', fixed_str)
            
            # Fix trailing commas in arrays
            fixed_str = re.sub(r',(\s*\])', r'\1', fixed_str)
            
            # Check if we have balanced brackets
            open_curly = fixed_str.count('{')
            close_curly = fixed_str.count('}')
            open_square = fixed_str.count('[')
            close_square = fixed_str.count(']')
            
            # Add missing closing brackets
            if open_curly > close_curly:
                fixed_str += '}' * (open_curly - close_curly)
            if open_square > close_square:
                fixed_str += ']' * (open_square - close_square)
            
            # Try to parse the fixed string
            try:
                json.loads(fixed_str)
                logger.info("Successfully fixed JSON string")
                return fixed_str
            except json.JSONDecodeError:
                logger.warning("Could not fix JSON automatically")
                return json_str

    @staticmethod
    def extract_json_from_text(text):
        """
        Extract a JSON object from text that might contain other content.
        Returns the extracted JSON string or None if no JSON-like structure is found.
        """
        # Try to find JSON object pattern
        json_match = re.search(r'(\{[\s\S]*\})', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            # Try to fix any JSON issues
            return PersonaAnalyzer.fix_json_string(json_str)
        return None

    @staticmethod
    def parse_analysis(response: str, selected_fields: Optional[List[str]] = None) -> Dict:
        """
        Parse the LLM response into a structured format.
        Now with improved JSON error handling.
        """
        if selected_fields is None:
            selected_fields = PERSONA_FIELDS
            
        def convert_to_string(value) -> str:
            """Convert various data types to string format"""
            if isinstance(value, (list, tuple)):
                return ", ".join(str(item) for item in value)
            elif isinstance(value, dict):
                return "; ".join(f"{k}: {v}" for k, v in value.items())
            return str(value)
            
        try:
            # First try to parse as is
            analysis = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            
            # Try to extract and fix JSON
            fixed_json = PersonaAnalyzer.extract_json_from_text(response)
            if fixed_json:
                try:
                    analysis = json.loads(fixed_json)
                    logger.info("Successfully extracted and fixed JSON")
                except json.JSONDecodeError as e2:
                    logger.error(f"Failed to parse extracted JSON: {e2}")
                    return {field: "Error: Invalid JSON response" for field in selected_fields}
            else:
                # Try direct fixing as a last resort
                fixed_json = PersonaAnalyzer.fix_json_string(response)
                try:
                    analysis = json.loads(fixed_json)
                    logger.info("Successfully fixed JSON directly")
                except json.JSONDecodeError:
                    logger.error("All JSON fixing attempts failed")
                    return {field: "Error: Invalid JSON response" for field in selected_fields}
        
        # Process the successfully parsed JSON
        result = {}
        for field in selected_fields:
            if field not in analysis:
                logger.warning(f"Missing required field: {field}")
                result[field] = "N/A"
                continue
            
            try:
                result[field] = convert_to_string(analysis[field])
                if not result[field].strip():
                    logger.warning(f"Field {field} is empty")
                    result[field] = "N/A"
            except Exception as e:
                logger.error(f"Error converting field {field}: {e}")
                result[field] = "N/A"
                
        return result

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


class ExtendedPersonaAnalyzer(PersonaAnalyzer):
    """
    Extended PersonaAnalyzer with additional prompt generation capabilities.
    """
    def create_persona_prompt(self, posts: List[Dict], conversations: List[Dict], 
                            n_posts: int, use_random_fields: bool = False, 
                            num_fields: int = 5) -> Union[str, Tuple[str, List[str]]]:
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
                
        prompt_template, selected_fields = self._prompt_generator.generate_prompt_version(num_fields)
        
        formatted_prompt = prompt_template.format(
            posts_text=posts_text,
            conversations_text=conversations_text
        )
        
        return formatted_prompt, selected_fields
        
    def analyze_persona_from_files(
        self, 
        posts_path: str, 
        conversations_path: Optional[str], 
        output_path: str, 
        n_posts: int = 5, 
        n_conversations: int = 5,
        use_random_fields: bool = False,
        num_fields: int = 5
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
                
                prompt_result = self.create_persona_prompt(
                    user_posts, 
                    user_conversations, 
                    n_posts,
                    use_random_fields=use_random_fields,
                    num_fields=num_fields
                )
                
                if use_random_fields:
                    prompt, selected_fields = prompt_result
                else:
                    prompt, selected_fields = prompt_result, PERSONA_FIELDS
                
                response = self.llm_client.call(prompt)
                persona_results[user] = self.parse_analysis(response, selected_fields)

            self.save_json(persona_results, output_path)
            logger.info(f"Persona analysis complete. Results saved to {output_path}")

        except Exception as e:
            logger.error("Failed to analyze personas:")
            logger.error(traceback.format_exc())
            raise