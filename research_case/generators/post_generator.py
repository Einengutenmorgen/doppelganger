from typing import Dict, Optional
import json
import logging
from dataclasses import dataclass

from research_case.analyzers.llm_client import LLMClient

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

{format_persona_section(PERSONA_FIELDS)}

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

Return a JSON object with this structure:
JSON Response: {{"post_text": "Can't believe my morning coffee costs $7 now 😤 Corporate greed is getting out of hand! Remember when it was $3? #inflation #ripoff"}}
"""

        prompt_text = template.format(
            **{field: prompt.persona[field] for field in PERSONA_FIELDS},
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
        template = """Given this social media post:
"{post}"

Create a brief, generic description of what kind of post this is and its general topic, 
WITHOUT including any specific details, names, or quotes from the original.
Keep it high-level and abstract.

Example input: "Just watched the new Spider-Man movie. The special effects were amazing! @TomHolland is the best Spider-Man ever #NWH"
Example output: A post about a recently watched superhero movie (Spiderman) and its lead actor

Write ONLY the generic description
Return a JSON object with this structure:
{{ "stimulus": "your generic description here" }}
Remember: Respond ONLY with the JSON object, nothing else.
"""


        try:
            response = self.llm_client.call(
                template.format(post=original_post),
                temperature=0.4,  # Lower temperature for more consistent outputs
                max_tokens=50
            )
            
            # Parse JSON response and extract stimulus
            response_json = json.loads(response)
            return response_json.get('stimulus', '')
            
        except Exception as e:
            logger.error(f"Error creating stimulus: {e}")
            raise