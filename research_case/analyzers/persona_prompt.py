# """Shared persona prompts and templates.

# Contains templates and prompts for:
# 1. Basic persona analysis (PERSONA_ANALYSIS_PROMPT)
# 2. Detailed character analysis (CHARACTER_ANALYSIS_PROMPT)
# """

# EXAMPLE_PERSONA = {
#     "writing_style": "Casual and direct writing style with medium-length sentences. Uses technical terminology when discussing professional topics while maintaining accessibility.",
#     "tone": "Professional yet approachable tone, showing enthusiasm for technical subjects while maintaining friendly demeanor.",
#     "topics": "Software development practices, team collaboration, technical mentorship, industry trends",
#     "personality_traits": "Analytical and detail-oriented, collaborative team player, patient mentor, pragmatic problem-solver",
#     "engagement_patterns": "Provides thorough and detailed responses, engages regularly with consistent patterns, maintains depth in technical discussions, takes mentor role in conversations",
#     "language_preferences": "Uses technical terms with clear explanations, communicates in structured and methodical way, frequently employs examples and analogies"
# }

# PERSONA_FIELDS = [
#     "writing_style",
#     "tone",
#     "topics",
#     "personality_traits",
#     "engagement_patterns",
#     "language_preferences"
# ]

# PERSONA_ANALYSIS_PROMPT = """Task: Analyze the following user-generated content to infer a detailed persona. Base all conclusions exclusively on the provided text samples.
# User Posts:
# {posts_text}
# {conversations_text}

# Provide a detailed analysis focusing on these key aspects:
# 1. Writing Style:
# - Describe formality level, sentence structure, and patterns
# - Include observations about vocabulary and writing techniques
# - Combine all observations into one coherent description

# 2. Tone:
# - Describe overall emotional tone and variations
# - Include balance of professional/casual elements
# - Combine all tone observations into one description

# 3. Topics:
# - List main discussion areas and interests
# - Include recurring themes
# - Combine all topics into one comma-separated description

# 4. Personality Traits:
# - Describe key personality characteristics
# - Include communication and behavioral traits
# - Combine all traits into one comma-separated description

# 5. Engagement Patterns:
# - Describe how they respond and interact
# - Include patterns of engagement and conversation depth
# - Combine all patterns into one coherent description

# 6. Language Preferences:
# - Describe vocabulary choices and expression methods
# - Include communication patterns and preferences
# - Combine all preferences into one coherent description

# FORMAT REQUIREMENT:
# Return a JSON object where ALL fields contain single string values.
# Combine multiple points into comma-separated lists or flowing descriptions.
# {EXAMPLE_PERSONA}

# Important Notes:
# - Use specific examples from the text to support observations
# - If there's insufficient data for any category, note this in that field
# - Aim for clear, detailed descriptions
# - Keep all responses as single strings"""

# Second prompt based on categroies selected by Gpt-01-preview
# EXAMPLE_PERSONA = {
#     "humor_style": "Dry and witty with frequent use of wordplay and subtle references, tends toward self-deprecating humor while maintaining intellectual depth",
#     "communication_patterns": "Articulate and verbose, favors complex sentence structures, regularly engages in detailed discussions with thorough explanations",
#     "emotional_expression": "Reserved yet authentic, shows empathy through thoughtful responses, expresses enthusiasm for intellectual topics",
#     "values_beliefs": "Values intellectual discourse, scientific thinking, ethical behavior, shows strong commitment to evidence-based reasoning",
#     "interests_hobbies": "Technology, programming, scientific developments, literature, philosophical discussions, exploring complex systems",
#     "social_dynamics": "Maintains respectful, professional relationships, acts as mentor figure, engages deeply with intellectual peers",
#     "coping_mechanisms": "Uses humor to diffuse tension, approaches problems analytically, seeks understanding through discussion",
#     "ethical_framework": "Strong emphasis on intellectual honesty, fairness, and ethical consideration in decision-making",
#     "aspirations": "Professional growth in technical fields, intellectual development, community building and knowledge sharing",
#     "cultural_context": "Influenced by academic/technical culture, shows awareness of global perspectives, values multicultural understanding",
#     "demographic_indicators": "Indicators suggest: professional in technical field, urban environment, higher education background",
#     "sociopolitical_leanings": "Evidence-based policy preferences, focuses on technical and practical solutions to social issues",
#     "length_of_post": "What is the sentence length? Short phrases or more explainable; full sentence half sentences"
# }

# PERSONA_FIELDS = [
#     "humor_style",
#     "communication_patterns",
#     "emotional_expression",
#     "values_beliefs",
#     "interests_hobbies",
#     "social_dynamics",
#     "coping_mechanisms",
#     "ethical_framework",
#     "aspirations",
#     "cultural_context",
#     "demographic_indicators",
#     "sociopolitical_leanings",
#     "length_of_post"
# ]

# PERSONA_ANALYSIS_PROMPT = """Task: Analyze the following social media posts to produce a detailed and comprehensive character description of the user. Base all conclusions exclusively on the provided content.

# Social Media Posts:
# {posts_text}
# {conversations_text}

# Provide a detailed analysis focusing on these key aspects:

# 1. Humor Style:
# - Analyze style of humor and wit
# - Note patterns in jokes and playful interactions
# - Combine observations into one coherent description

# 2. Communication Patterns:
# - Examine writing style and communication preferences
# - Note interaction patterns and discussion approaches
# - Combine all patterns into one description

# 3. Emotional Expression:
# - Analyze how emotions are conveyed
# - Note emotional range and depth
# - Combine observations into one flowing description

# 4. Values and Beliefs:
# - Identify core values and belief systems
# - Note philosophical and ideological patterns
# - Combine into one comprehensive description

# 5. Interests and Hobbies:
# - List main areas of interest and engagement
# - Note depth and patterns of engagement
# - Combine into one detailed description

# 6. Social Dynamics:
# - Analyze interaction patterns and relationships
# - Note social roles and behavioral patterns
# - Combine into one coherent description

# 7. Coping Mechanisms:
# - Identify stress response patterns
# - Note problem-solving approaches
# - Combine into one flowing description

# 8. Ethical Framework:
# - Analyze moral and ethical viewpoints
# - Note decision-making patterns
# - Combine into one comprehensive description

# 9. Aspirations:
# - Identify goals and ambitions
# - Note patterns in future-oriented thinking
# - Combine into one detailed description

# 10. Cultural Context:
# - Analyze cultural and societal influences
# - Note cultural references and perspectives
# - Combine into one flowing description

# 11. Demographic Indicators:
# - Estimate basic demographic information
# - Support with specific evidence
# - Combine into one careful description

# 12. Sociopolitical Leanings:
# - Identify political and social associations
# - Note patterns in societal views
# - Combine into one balanced description

# 13. Length of post:
# - Analyze typical post length (character count, word count, paragraph structure)
# - Examine sentence completeness and complexity patterns
# - Note structural variations across different types of posts
# - Identify linguistic style (formal/informal, spoken/written patterns)
# - Document temporal patterns in post length
# - Analyze relationship between content type and post length
# - Note any contextual factors affecting length choices
# - Combine observations into one comprehensive description

# FORMAT REQUIREMENT:
# Return a JSON object where ALL fields contain single string values.
# Combine multiple points into flowing descriptions.
# {EXAMPLE_PERSONA}

# Important Notes:
# - Base all conclusions solely on provided content
# - Support observations with specific examples
# - Note if insufficient data for any category
# - Avoid unwarranted assumptions
# - Keep descriptions detailed but objective"""






# """
#Run Number #03 
# append just a lot of fields
# """
EXAMPLE_PERSONA = {
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

PERSONA_FIELDS = [
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

PERSONA_ANALYSIS_PROMPT = """Task: Analyze the following social media posts to produce a detailed and comprehensive character description of the user. Base all conclusions exclusively on the provided content.

Social Media Posts:
{posts_text}
{conversations_text}

Provide a detailed analysis focusing on these key aspects:

1. Brevity Style:
- Analyze how they work within character limits
- Note patterns in message length and structure
- Combine observations into one coherent description

2. Language Formality:
- Examine formal vs casual language usage
- Note context-specific language patterns
- Combine into one flowing description

3. Narrative Voice:
- Analyze personal vs professional tone
- Note voice consistency and variations
- Combine into one comprehensive description

4. Vocabulary Range:
- Assess language diversity and complexity
- Note specialized terminology usage
- Combine into one detailed description

5. Punctuation Style:
- Examine punctuation patterns and preferences
- Note how punctuation affects meaning
- Combine into one coherent description

6. Controversy Handling:
- Analyze approach to sensitive topics
- Note conflict management strategies
- Combine into one flowing description

7. Community Role:
- Identify their position in communities
- Note interaction patterns with others
- Combine into one comprehensive description

8. Content Triggers:
- Analyze what prompts them to post
- Note response patterns to events
- Combine into one detailed description

9. Reaction Patterns:
- Examine responses to events/news
- Note timing and tone of reactions
- Combine into one coherent description

10. Message Effectiveness:
- Analyze how well points are conveyed
- Note engagement and clarity patterns
- Combine into one flowing description

11. Opinion Expression:
- Examine how viewpoints are shared
- Note argumentation patterns
- Combine into one comprehensive description

12. Emotional Expression:
- Analyze how feelings are conveyed
- Note emotional range and depth
- Combine into one detailed description

13. Cognitive Patterns:
- Examine thinking styles in posts
- Note problem-solving approaches
- Combine into one coherent description

14. Social Orientation:
- Analyze community vs individual focus
- Note relationship building patterns
- Combine into one flowing description

15. Conflict Approach:
- Examine handling of disagreements
- Note de-escalation strategies
- Combine into one comprehensive description

16. Value Signals:
- Identify emphasized principles
- Note value expression patterns
- Combine into one detailed description

17. Identity Projection:
- Analyze self-presentation style
- Note consistency in identity expression
- Combine into one coherent description

18. Belief Expression:
- Examine how convictions are shared
- Note belief defense patterns
- Combine into one flowing description

19. Stress Indicators:
- Analyze behavior during pressure
- Note changes in communication style
- Combine into one comprehensive description

20. Adaptability Signs:
- Examine response to platform changes
- Note flexibility in communication
- Combine into one detailed description

21. Authenticity Markers:
- Analyze genuineness in communication
- Note consistency across contexts
- Combine into one coherent description

FORMAT REQUIREMENT:
Return a JSON object where ALL fields contain single string values.
Combine multiple points into flowing descriptions.
{EXAMPLE_PERSONA}

Important Notes:
- Base all conclusions solely on provided content
- Support observations with specific examples
- Note if insufficient data for any category
- Avoid unwarranted assumptions
- Keep descriptions detailed but objective"""