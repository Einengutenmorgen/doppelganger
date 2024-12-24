import json
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import traceback2 as traceback

from .llm_client import LLMClient

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
            conversations = self.load_json(conversations_path) if conversations_path else None

            # Step 2: Analyze each user's persona
            persona_results = {}
            for user, user_posts in posts.items():
                user_conversations = (
                    self.get_user_conversations(user, conversations, n_conversations)
                    if conversations else [{}]
                )
                prompt = self.create_persona_prompt(user_posts, user_conversations, n_posts)
                response = self.llm_client.call(prompt)
                persona_results[user] = self.parse_analysis(response)

            # Step 3: Save output
            self.save_json(persona_results, output_path)
            logger.info(f"Persona analysis complete. Results saved to {output_path}")

        except Exception as e:
            logger.error("Failed to analyze personas:")
            logger.error(traceback.format_exc())
            
            
    def create_persona_prompt(
        self, 
        posts: List[Dict], 
        conversations: List[Dict], 
        n_posts: int
    ) -> str:
        """
        Create a prompt for persona analysis based on posts and conversations.
        Returns a prompt that encourages detailed analysis in a simple format.
        """
        # Extract top `n` posts
        posts_text = "\n".join(
            f"{i+1}. {post['full_text']}" for i, post in enumerate(posts[:n_posts])
        )

        # Extract conversations if present
        conversations_text = ""
        if conversations:
            conversations_text = "\n".join(
                f"{i+1}. {conv['full_text']}" for i, conv in enumerate(conversations[:n_posts])
            )
            conversations_text = "\nUser Conversations:\n" + conversations_text

        example_output = {
            "writing_style": "Casual and direct writing style with medium-length sentences. Uses technical terminology when discussing professional topics while maintaining accessibility.",
            "tone": "Professional yet approachable tone, showing enthusiasm for technical subjects while maintaining friendly demeanor.",
            "topics": "Software development practices, team collaboration, technical mentorship, industry trends",
            "personality_traits": "Analytical and detail-oriented, collaborative team player, patient mentor, pragmatic problem-solver",
            "engagement_patterns": "Provides thorough and detailed responses, engages regularly with consistent patterns, maintains depth in technical discussions, takes mentor role in conversations",
            "language_preferences": "Uses technical terms with clear explanations, communicates in structured and methodical way, frequently employs examples and analogies"
        }

        return f"""Task: Analyze the following user-generated content to infer a detailed persona. Base all conclusions exclusively on the provided text samples.

    User Posts:
    {posts_text}
    {conversations_text}

    Provide a detailed analysis focusing on these key aspects:

    1. Writing Style:
    - Describe formality level, sentence structure, and patterns
    - Include observations about vocabulary and writing techniques
    - Combine all observations into one coherent description

    2. Tone:
    - Describe overall emotional tone and variations
    - Include balance of professional/casual elements
    - Combine all tone observations into one description

    3. Topics:
    - List main discussion areas and interests
    - Include recurring themes
    - Combine all topics into one comma-separated description

    4. Personality Traits:
    - Describe key personality characteristics
    - Include communication and behavioral traits
    - Combine all traits into one comma-separated description

    5. Engagement Patterns:
    - Describe how they respond and interact
    - Include patterns of engagement and conversation depth
    - Combine all patterns into one coherent description

    6. Language Preferences:
    - Describe vocabulary choices and expression methods
    - Include communication patterns and preferences
    - Combine all preferences into one coherent description

    FORMAT REQUIREMENT:
    Return a JSON object where ALL fields contain single string values.
    Combine multiple points into comma-separated lists or flowing descriptions.

    Example of the required format:
    {json.dumps(example_output, indent=2)}

    Important Notes:
    - Use specific examples from the text to support observations
    - If there's insufficient data for any category, note this in that field
    - Aim for clear, detailed descriptions
    - Keep all responses as single strings"""

    @staticmethod
    def parse_analysis(response: str) -> Dict:
        """
        Parse the LLM response into a structured analysis dictionary.
        Converts all values to strings for simplified processing.
        """
        try:
            # Parse JSON response
            analysis = json.loads(response)
            
            # Required fields
            required_fields = [
                "writing_style",
                "tone",
                "topics",
                "personality_traits",
                "engagement_patterns",
                "language_preferences"
            ]
            
            def convert_to_string(value) -> str:
                """Convert any value to a string representation."""
                if isinstance(value, (list, tuple)):
                    return ", ".join(str(item) for item in value)
                elif isinstance(value, dict):
                    return "; ".join(f"{k}: {v}" for k, v in value.items())
                return str(value)
            
            # Process each field
            result = {}
            for field in required_fields:
                if field not in analysis:
                    raise KeyError(f"Missing required field: {field}")
                result[field] = convert_to_string(analysis[field])
                
                # Ensure non-empty content
                if not result[field].strip():
                    raise ValueError(f"Field {field} cannot be empty")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid analysis structure: {e}")
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