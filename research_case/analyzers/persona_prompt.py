"""Shared persona prompts and templates."""

EXAMPLE_PERSONA = {
    "writing_style": "Casual and direct writing style with medium-length sentences. Uses technical terminology when discussing professional topics while maintaining accessibility.",
    "tone": "Professional yet approachable tone, showing enthusiasm for technical subjects while maintaining friendly demeanor.",
    "topics": "Software development practices, team collaboration, technical mentorship, industry trends",
    "personality_traits": "Analytical and detail-oriented, collaborative team player, patient mentor, pragmatic problem-solver",
    "engagement_patterns": "Provides thorough and detailed responses, engages regularly with consistent patterns, maintains depth in technical discussions, takes mentor role in conversations",
    "language_preferences": "Uses technical terms with clear explanations, communicates in structured and methodical way, frequently employs examples and analogies"
}

PERSONA_FIELDS = [
    "writing_style",
    "tone", 
    "topics",
    "personality_traits",
    "engagement_patterns",
    "language_preferences"
]

PERSONA_ANALYSIS_PROMPT = """Task: Analyze the following user-generated content to infer a detailed persona. Base all conclusions exclusively on the provided text samples.

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

{EXAMPLE_PERSONA}

Important Notes:
- Use specific examples from the text to support observations
- If there's insufficient data for any category, note this in that field
- Aim for clear, detailed descriptions
- Keep all responses as single strings"""