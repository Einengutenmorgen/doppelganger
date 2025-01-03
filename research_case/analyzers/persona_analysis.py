import json
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import traceback2 as traceback

from .llm_client import LLMClient
from .persona_prompt import PERSONA_FIELDS, PERSONA_ANALYSIS_PROMPT, EXAMPLE_PERSONA

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
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def analyze_persona_from_files(
        self, 
        posts_path: str, 
        conversations_path: Optional[str], 
        output_path: str, 
        n_posts: int = 5, 
        n_conversations: int = 5
    ) -> None:
        """
        Main entry point for persona analysis from file inputs.
        Args:
            posts_path (str): Path to JSON file with user posts.
            conversations_path (Optional[str]): Path to JSON file with conversations.
            output_path (str): Path to save the persona analysis results.
            n_posts (int): Number of posts to include in the analysis.
            n_conversations (int): Number of conversations to include in the analysis.
        """
        try:
            # Step 1: Load inputs
            posts = self.load_json(posts_path)
            conversations = None
            if conversations_path and os.path.exists(conversations_path):
                    conversations = self.load_json(conversations_path)
                    
            # Step 2: Analyze each user's persona
            persona_results = {}
            for user, user_posts in posts.items():
                user_conversations = []  # Default empty list if no conversations
                if conversations:
                    user_conversations = self.get_user_conversations(user, conversations, n_conversations)
                prompt = self.create_persona_prompt(user_posts, user_conversations, n_posts)
                response = self.llm_client.call(prompt)
                persona_results[user] = self.parse_analysis(response)


            # Step 3: Save output
            self.save_json(persona_results, output_path)
            logger.info(f"Persona analysis complete. Results saved to {output_path}")

        except Exception as e:
            logger.error("Failed to analyze personas:")
            logger.error(traceback.format_exc())
            
            
    def create_persona_prompt(self, posts: List[Dict], conversations: List[Dict], n_posts: int) -> str:
        # Extract top `n` posts
        posts_text = "\n".join(
            f"{i+1}. {post['full_text']}" for i, post in enumerate(posts[:n_posts])
        )

        # Extract conversations if present
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


    @staticmethod
    def parse_analysis(response: str) -> Dict:
        """Parse the LLM response into a structured analysis dictionary."""
        try:
            analysis = json.loads(response)
            
            def convert_to_string(value) -> str:
                if isinstance(value, (list, tuple)):
                    return ", ".join(str(item) for item in value)
                elif isinstance(value, dict):
                    return "; ".join(f"{k}: {v}" for k, v in value.items())
                return str(value)
            
            result = {}
            for field in PERSONA_FIELDS:
                if field not in analysis:
                    raise KeyError(f"Missing required field: {field}")
                result[field] = convert_to_string(analysis[field])
                
                if not result[field].strip():
                    raise ValueError(f"Field {field} cannot be empty")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise

    @staticmethod
    def load_json(file_path: str) -> Union[Dict, List]:
        """
        Load a JSON file and return its content.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_json(data: Dict, file_path: str) -> None:
        """
        Save a dictionary as a JSON file.
        """
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def get_user_conversations(user: str, conversations: Union[Dict, List[Dict]], n_conversations: int) -> List[Dict]:
        """
        Extract top `n` conversations for a user from the conversation data.
        
        Args:
            user: User identifier
            conversations: Either a dict mapping users to their conversations,
                        or a list of conversation dictionaries
            n_conversations: Number of conversations to return
            
        Returns:
            List of conversation dictionaries for the user
        """
        if conversations is None:
            return []
            
        # Case 1: If conversations is a dict (user -> conversations mapping)
        if isinstance(conversations, dict):
            user_convs = conversations.get(user, [])
            if not isinstance(user_convs, list):
                logger.error(f"Expected list of user conversations, got {type(user_convs)}")
                return []
                
        # Case 2: If conversations is a list of conversation dicts
        elif isinstance(conversations, list):
            user_convs = [
                conv for conv in conversations 
                if any(participant.get('id') == user for participant in conv.get('participants', []))
            ]
            
        else:
            logger.error(f"Expected dict or list of conversations, but got {type(conversations)}")
            return []

        # Sort conversations by reply count or other criteria if available
        user_convs.sort(key=lambda x: x.get("reply_count", 0), reverse=True)
        return user_convs[:n_conversations]