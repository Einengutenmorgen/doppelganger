from typing import Dict, Optional, List
import json
import logging
from dataclasses import dataclass

from research_case.analyzers.llm_client import LLMClient
from research_case.analyzers.persona_prompt import PERSONA_FIELDS, PERSONA_ANALYSIS_PROMPT, EXAMPLE_PERSONA

logger = logging.getLogger(__name__)

@dataclass
class GenerationPrompt:
    """Container for post generation inputs"""
    persona: Dict[str, str]
    stimulus: str
    context: Optional[str] = None

def format_persona_section(fields: List[str]) -> str:
    """Generate the persona characteristics section of the prompt."""
    return "\n".join(f"{field.replace('_', ' ').title()}: {{{field}}}" 
                    for field in fields)


class PostGenerator:
    """Generator for creating synthetic social media posts based on personas"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def _format_persona_section(self, persona: Dict[str, str]) -> str:
        """Format persona characteristics based on PERSONA_FIELDS."""
        sections = []
        for field in PERSONA_FIELDS:
            if field not in persona:
                raise KeyError(f"Missing required persona field: {field}")
            formatted_field = field.replace('_', ' ').title()
            sections.append(f"{formatted_field}: {persona[field]}")
        return "\n".join(sections)
    
    def generate_post(self, prompt: GenerationPrompt) -> str:
        """
        Generate a social media post based on a persona and stimulus.
        
        Args:
            prompt: GenerationPrompt containing persona and stimulus
            
        Returns:
            Generated post text
        """
        for field in PERSONA_FIELDS:
            if field not in prompt.persona:
                raise KeyError(f"Missing required persona field: {field}")

        template = """You are a social media user with the following characteristics:

{persona_characteristics}

Context: You are writing a social media post in response to the following stimulus:
{stimulus}

Task: Write ONE social media post that this persona would create in response to the stimulus. 
The post should reflect the persona's writing style, tone, and typical patterns.

Important guidelines:
- Stay true to the persona's characteristics
- Make it feel authentic and natural
- Include typical social media elements (hashtags, @mentions) if that fits the persona
- Keep the length realistic for social media
- Do not directly copy or closely paraphrase the stimulus
- Formulate precise and 

Return a JSON object with this structure:
JSON Response: {{"post_text": "Can't believe my morning coffee costs $7 now 😤 Corporate greed is getting out of hand!"}}
"""

        prompt_text = template.format(
            persona_characteristics=self._format_persona_section(prompt.persona),
            stimulus=prompt.stimulus
        )

        try:
            response = self.llm_client.call(
                prompt_text,
                temperature=0.7,  # Use moderate temperature for creativity
                max_tokens=100
            )
            
            # Clean up the response
            generated_post = response.strip().strip('"').strip()
            
            return generated_post
            
        except Exception as e:
            logger.error(f"Error generating post: {e}")
            raise

class StimulusGenerator:
    """Generates generic stimuli from original posts for testing generation"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def create_post_stimulus(self, original_post: str) -> str:
        """
        Create a generic stimulus from an original post.
        Removes specific details while preserving the general topic/intent.
        
        Args:
            original_post: The original post text
            
        Returns:
            Generic stimulus description
        """
        template = """Describe the topic of the tweet with enough detail so that it can be reused to create a similar tweet. 
It is important that you extract the topic of the tweet and phrase it as neutrally as possible, completely removing any original opinion, viewpoint, or commentary, while retaining important details and facts.
Includde the style of the tweet (e.g. question, comment, opinion, statement, information etc.) before the topic. The style should also not give away the direction of the tweet.
If the tweet requires you to invent or create a context, please reply with 'CONTEXT MISSING'.
Don't add any additional remarks or comments.

Tweet:
"{post}"

Respond ONLY with the stimulus description, nothing else.

"""


        try:
            response = self.llm_client.call(
                template.format(post=original_post),
                temperature=0.2,  # Lower temperature for more consistent outputs
                max_tokens=50,
                response_format=None
            )
            stimulus = response.strip().strip('"').strip()
            
            # Handle context missing case
            if stimulus.upper() == 'CONTEXT MISSING':
                logger.warning(f"Context missing for tweet: {original_post[:50]}...")
                return stimulus
                
            return stimulus
        except Exception as e:
            logger.error(f"Error creating stimulus: {e}")
            raise