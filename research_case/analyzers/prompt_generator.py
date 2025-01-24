from typing import List, Dict, Tuple, Optional
import random
import json
import logging
from .persona_prompt import PERSONA_FIELDS, EXAMPLE_PERSONA

logger = logging.getLogger(__name__)

def extend_persona_analyzer(PersonaAnalyzer):
        """
        Factory function to extend PersonaAnalyzer with prompt generation capabilities
        without modifying the original class.
        """
        original_create_persona_prompt = PersonaAnalyzer.create_persona_prompt
        
        def create_persona_prompt(self, posts: List[Dict], conversations: List[Dict], n_posts: int, 
                                use_random_fields: bool = False, num_fields: int = 5) -> str:
            if not use_random_fields:
                # Use original implementation
                return original_create_persona_prompt(self, posts, conversations, n_posts)
                
            # Extract posts and conversations text
            posts_text = "\n".join(
                f"{i+1}. {post['full_text']}" for i, post in enumerate(posts[:n_posts])
            )

            conversations_text = ""
            if conversations and len(conversations) > 0:
                conversations_text = "\n".join(
                    f"{i+1}. {conv['full_text']}" for i, conv in enumerate(conversations[:n_posts])
                )
                conversations_text = "\nUser Conversations:\n" + conversations_text

            # Use generator for random fields version
            if not hasattr(self, '_prompt_generator'):
                self._prompt_generator = PersonaPromptGenerator(
                    base_prompt="",  # Base prompt is built into the generator
                    example_persona=EXAMPLE_PERSONA,
                    all_fields=PERSONA_FIELDS
                )
                
            prompt, selected_fields = self._prompt_generator.generate_prompt_version(num_fields)
            return prompt.format(posts_text=posts_text, conversations_text=conversations_text)

        # Attach the new method to the class
        PersonaAnalyzer.create_persona_prompt = create_persona_prompt
        return PersonaAnalyzer


class PersonaPromptGenerator:
    """Generator for creating different versions of the persona analysis prompt."""
    
    def __init__(self, base_prompt: str, example_persona: Dict[str, str], all_fields: List[str]):
        self.base_prompt = base_prompt
        self.example_persona = example_persona
        self.all_fields = all_fields

    def generate_prompt_version(self, num_categories: int = 5) -> Tuple[str, List[str]]:
        try:
            if num_categories <= 0 or num_categories > len(self.all_fields):
                raise ValueError(f"num_categories must be between 1 and {len(self.all_fields)}")
                
            selected_fields = random.sample(self.all_fields, num_categories)
            logger.debug(f"Selected fields: {selected_fields}")
            
            pruned_example = {
                field: self.example_persona[field]
                for field in selected_fields
                if field in self.example_persona  # Add this check
            }
            
            analysis_sections = []
            for i, field in enumerate(selected_fields, 1):
                field_title = field.replace('_', ' ').title()
                section = f"{i}. {field_title}:\n"
                section += f"- Analyze {self._get_analysis_instruction(field)}\n"
                section += "- Note patterns and variations\n"
                section += "- Combine into one coherent description\n"
                section += f"- Required format: \"{field}\": \"<description>\"\n"  # Add format reminder
                analysis_sections.append(section)
            
            prompt = self._create_prompt_text(
                analysis_sections="\n".join(analysis_sections),
                example_persona=pruned_example,
                selected_fields=selected_fields
            )
            
            return prompt, selected_fields
        except Exception as e:
            logger.error(f"Error generating prompt version: {e}")
            raise

    def _get_analysis_instruction(self, field: str) -> str:
        instructions = {
            "brevity_style": "how they work within character limits",
            "language_formality": "level of formal vs casual language",
            "narrative_voice": "personal vs professional tone",
            "vocabulary_range": "diversity of language used",
            "punctuation_style": "distinctive punctuation patterns",
            "controversy_handling": "approach to sensitive topics",
            "community_role": "position in their Twitter communities",
            "content_triggers": "what prompts them to post",
            "reaction_patterns": "how they respond to events/news",
            "message_effectiveness": "how well they convey their points",
            "opinion_expression": "how they share viewpoints",
            "emotional_expression": "how they convey feelings",
            "cognitive_patterns": "thinking styles revealed in posts",
            "social_orientation": "individual vs community focus",
            "conflict_approach": "how they handle disagreements",
            "value_signals": "what principles they emphasize",
            "identity_projection": "how they present themselves",
            "belief_expression": "how they share convictions",
            "stress_indicators": "patterns during high-stress periods",
            "adaptability_signs": "how they handle platform changes",
            "authenticity_markers": "genuineness in communication"
        }
        return instructions.get(field, "this aspect of their communication")

    def _create_prompt_text(self, analysis_sections: str, example_persona: Dict[str, str], selected_fields: List[str]) -> str:
        
        # Escape any curly braces in the example JSON by doubling them
        example_json = json.dumps(example_persona, indent=4).replace("{", "{{").replace("}", "}}")
        fields_list = "\n".join(f"- {field}" for field in selected_fields)

        
        return f"""Task: Analyze the following social media posts to produce a detailed character description of the user. Base all conclusions exclusively on the provided content.
    
    Social Media Posts:
    {{posts_text}}
    {{conversations_text}}

    Provide a detailed analysis focusing on these key aspects:

    {analysis_sections}
    
    REQUIRED FIELDS (must include ALL of these):
    {fields_list}

    FORMAT REQUIREMENT:
    Return a JSON object where ALL fields contain single string values.
    Combine multiple points into flowing descriptions.
    {example_json}

    Important Notes:
    - Base all conclusions solely on provided content
    - Avoid unwarranted assumptions
    - Keep descriptions detailed but objective
    - Return ONLY a JSON object with the required fields"""



    