EXAMPLE_FIELD_PERSONA = {
    "general_decription": "A sports commentator mainly for football, active for several news outlets, properly Minnesota fan, Male mid 30s from the midwest",
    
    "brevity_style": "Favors concise, impactful statements under 180 characters; uses abbreviations sparingly; breaks longer thoughts into well-structured threads of 3-4 tweets",
    
    "language_formality": "Mixed formality level - uses professional language for technical discussions but adopts casual, conversational tone for community engagement. Frequently includes industry terminology with accompanying explanations",
    
    "narrative_voice": "Primarily professional with strategic personal anecdotes. Maintains expert positioning while sharing occasional glimpses of personal life through relevant stories",
    
    "vocabulary_range": "Extensive technical vocabulary with comfortable use of academic terminology. Regularly employs metaphors to explain complex concepts while avoiding trending slang",
    
    "punctuation_style": "Strategic use of semicolons and em-dashes for asides. Minimal exclamation points. Uses ellipsis to create suspense in threads. Careful with parenthetical statements",
    
    "controversy_handling": "Addresses controversies with data-backed arguments. Diplomatically redirects heated discussions to merit-based analysis. Uses 'agree to disagree' approach when needed",
    
    "community_role": "Acts as respected authority and frequent mentor. Bridges gaps between different community segments. Known for constructive contributions and tension diffusion",
    
    "content_triggers": "Responds to major announcements, common misconceptions, requests for help, new research developments, and community members seeking guidance",
    
    "reaction_patterns": "Verifies information before responding. Waits for official sources on controversies. Provides balanced perspective on changes. Offers solutions over criticism",
    
    "message_effectiveness": "Generates high engagement on educational content. Posts frequently bookmarked and cited. Receives consistent positive feedback on clarity of communication",
    
    "opinion_expression": "Presents viewpoints with supporting evidence. Acknowledges expertise limitations. Uses experience qualifiers. Supports controversial positions with data",
    
    "emotional_expression": "Reserved but authentic emotional display. Shows enthusiasm for achievements. Expresses empathy for struggles. Uses humor to lighten complex topics",
    
    "cognitive_patterns": "Systematic problem-solver who breaks down complex topics. Frequently uses analogies. Builds arguments from first principles. Shows clear logical progression",
    
    "social_orientation": "Community-focused while maintaining professional boundaries. Actively mentors others. Shares credit generously. Promotes others' achievements. Builds consensus",
    
    "conflict_approach": "De-escalates through data and evidence. Acknowledges valid opposing points. Moves heated discussions private. Focuses on solutions rather than blame",
    
    "value_signals": "Emphasizes continuous learning, quality standards, community well-being, ethical development practices, support systems, and knowledge sharing",
    
    "identity_projection": "Positions self as experienced practitioner and lifelong learner. Maintains professional image while showing humanity. Emphasizes teaching and mentorship",
    
    "belief_expression": "Openly advocates for best practices, community health, ethical standards, and inclusivity. Supports beliefs with combination of experience and research",
    
    "stress_indicators": "Shows increased verbosity during high-pressure periods. More frequent self-corrections under stress. Shifts to more formal language in tense situations",
    
    "adaptability_signs": "Quickly adopts new features. Adapts communication style based on feedback. Evolves content format with platform changes. Shows flexibility in approach",
    
    "authenticity_markers": "Maintains consistent voice across topics. Admits mistakes publicly. Shares both successes and failures. Shows personality alignment in public and private"
}

PERSONA_FIELDS_F = [
    "general_decription",
    "brevity_style",           # How they work within character limits
    "language_formality",      # Level of formal vs casual language
    "narrative_voice",         # Personal vs professional tone
    "vocabulary_range",        # Diversity of language used
    "punctuation_style",       # Distinctive punctuation patterns
    "controversy_handling",    # Approach to sensitive topics
    "community_role",          # Position in their Twitter communities
    "content_triggers",        # What prompts them to post
    "reaction_patterns",       # How they respond to events/news
    "message_effectiveness",   # How well they convey their points
    "opinion_expression",      # How they share viewpoints
    "emotional_expression",    # How they convey feelings
    "cognitive_patterns",      # Thinking styles revealed in posts
    "social_orientation",      # Individual vs community focus
    "conflict_approach",       # How they handle disagreements
    "value_signals",          # What principles they emphasize
    "identity_projection",     # How they present themselves
    "belief_expression",       # How they share convictions
    "stress_indicators",       # Patterns during high-stress periods
    "adaptability_signs",      # How they handle platform changes
    "authenticity_markers"     # Genuineness in communication
]

NEW_PERSONA_FIELD_ANALYSIS_PROMPT = """
Task: Analyze the following social media posts to produce present categories that lists the names of all categories mentioned. Base all conclusions exclusively on the provided content.

For each of the following categories, if there is sufficient data in the provided content, provide a detailed analysis as a single string. If there is insufficient data for a category, do not include that category in your output.


Social Media Posts:
{posts_text}
{conversations_text}

Categories to consider:
1. Brevity Style:
   - Examine character limits, message length patterns, and structure.
2. Language Formality:
   - Evaluate the use of formal vs. casual language and any context-specific patterns.
3. Narrative Voice:
   - Assess whether the tone is personal, professional, or varies, and note consistency.
4. Vocabulary Range:
   - Analyze language diversity, complexity, and any specialized terminology.
5. Punctuation Style:
   - Investigate punctuation patterns and how they affect the conveyed meaning.
6. Controversy Handling:
   - Review the approach to sensitive topics and conflict management strategies.
7. Community Role:
   - Identify the user's role within communities and interaction patterns.
8. Content Triggers:
   - Determine what prompts the user to post and how they respond to events.
9. Reaction Patterns:
   - Examine timing and tone of responses to events or news.
10. Message Effectiveness:
    - Evaluate how clearly and engagingly the user conveys their points.
11. Opinion Expression:
    - Analyze how viewpoints are shared and the patterns of argumentation.
12. Emotional Expression:
    - Assess the range and depth of emotions conveyed.
13. Cognitive Patterns:
    - Explore the user’s thinking styles and problem-solving approaches.
14. Social Orientation:
    - Determine the focus on community versus individual interests and relationship building.
15. Conflict Approach:
    - Investigate how disagreements are handled and what de-escalation strategies are used.
16. Value Signals:
    - Identify any expressed principles or values.
17. Identity Projection:
    - Analyze self-presentation and the consistency of identity expression.
18. Belief Expression:
    - Examine how convictions are communicated and defended.
19. Stress Indicators:
    - Look for behavioral changes or shifts in communication under pressure.
20. Adaptability Signs:
    - Evaluate flexibility in communication and responses to platform changes.
21. Authenticity Markers:
    - Determine the genuineness and consistency in communication across contexts.


FORMAT REQUIREMENT:
Return a JSON object similar to the example below (only include keys for categories with sufficient data):
{EXAMPLE_PERSONA}

Instructions:
- Base all observations solely on the provided content.
- If there is insufficient data for a category, do not include that category in your output.
- ınclude a key "present_categories" that lists, as a comma-separated string, all the category names for which analysis was provided.
- Return your output as a comma-separated string (for "present_categories").
- Support observations with specific examples from the content where possible, and avoid unwarranted assumptions.

"""
