import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from research_case.analyzers.llm_client import LLMClient

logger = logging.getLogger(__name__)

@dataclass 
class ConversationPrompt:
    """Container for conversation generation inputs"""
    persona: Dict[str, str]
    conversation_history: List[Dict]
    parent_message: str
    context: Optional[str] = None

class ConversationGenerator:
    """Generator for creating synthetic conversation responses based on personas"""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize with LLM client"""
        self.llm_client = llm_client

    def generate_response(self, prompt: ConversationPrompt) -> str:
        """
        Generate a conversation response based on persona and context.
        
        Args:
            prompt: ConversationPrompt containing persona and conversation context
            
        Returns:
            Generated response text
        """
        template = """You are responding in a conversation thread as a social media user with these characteristics:

Writing Style: {writing_style}
Tone: {tone}
Topics of Interest: {topics}
Personality Traits: {personality_traits}
Typical Engagement Patterns: {engagement_patterns}
Language Preferences: {language_preferences}

Conversation Context:
Previous message you're responding to: {parent_message}

Full conversation history:
{history}

Task: Write a single social media response (maximum 280 characters) that this persona would write in reply to the previous message.

Your response should:
- Stay authentic to the persona's style and characteristics
- Feel natural and conversational 
- Include typical social media elements if that fits the persona
- Maintain consistent tone throughout
- Actually engage with the content of the message being replied to
- Keep the length realistic for social media
- Build on the existing conversation thread

Return response in JSON format:
{{ "response": "your response here" }}
"""

        prompt_text = template.format(
            writing_style=prompt.persona["writing_style"],
            tone=prompt.persona["tone"], 
            topics=prompt.persona["topics"],
            personality_traits=prompt.persona["personality_traits"],
            engagement_patterns=prompt.persona["engagement_patterns"],
            language_preferences=prompt.persona["language_preferences"],
            parent_message=prompt.parent_message,
            history=self._format_history(prompt.conversation_history)
        )

        try:
            response = self.llm_client.call(
                prompt_text,
                temperature=0.7,
                max_tokens=100
            )
            
            # Parse JSON response
            response_json = json.loads(response)
            return response_json.get('response', '')
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _format_history(self, history: List[Dict]) -> str:
        """Format conversation history for prompt"""
        formatted = []
        for msg in history:
            formatted.append(f"@{msg.get('screen_name', 'user')}: {msg.get('full_text', '')}")
        return "\n".join(formatted)