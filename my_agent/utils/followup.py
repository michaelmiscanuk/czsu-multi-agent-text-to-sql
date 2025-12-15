"""Follow-up Prompt Generation for CZSU Multi-Agent Text-to-SQL Application

This module provides functionality for generating follow-up prompt suggestions that help users
explore Czech Statistical Office (CZSU) data. It offers two generation strategies:

1. Template-based generation: Fast, deterministic prompts using predefined templates
2. AI-based generation: Creative, context-aware prompts using Azure OpenAI

The generation strategy is selected via the FOLLOWUP_PROMPTS_STRATEGY environment variable.

Functions:
=========
- generate_initial_followup_prompts(): Main entry point, selects strategy based on env config
- generate_initial_followup_prompts_template(): Template-based generation (default)
- generate_initial_followup_prompts_ai(): AI-based generation using GPT-4o-mini

Configuration:
=============
Environment Variables:
- FOLLOWUP_PROMPTS_STRATEGY: "template" (default) or "ai"
  - "template": Uses predefined templates with random selections (fast, no API calls)
  - "ai": Uses Azure OpenAI GPT-4o-mini for creative generation (slower, requires API)

Usage Examples:
==============
```python
# Default usage (respects FOLLOWUP_PROMPTS_STRATEGY env variable)
from my_agent.utils.followup import generate_initial_followup_prompts

prompts = generate_initial_followup_prompts()
# Returns 5 prompts: either from templates or AI based on configuration

# Force template-based generation
from my_agent.utils.followup import generate_initial_followup_prompts_template

prompts = generate_initial_followup_prompts_template()
# Always returns template-based prompts

# Force AI-based generation
from my_agent.utils.followup import generate_initial_followup_prompts_ai

prompts = generate_initial_followup_prompts_ai()
# Always returns AI-generated prompts
```

Template-based Generation:
==========================
Characteristics:
- Fast execution (no API calls)
- Deterministic with pseudo-random variety
- 15+ diverse prompt templates
- Multiple topic pools for placeholder substitution
- Ensures 5 unique prompts per call

Template categories:
- Population and demographic queries
- Economic indicators (GDP, wages, employment)
- Regional comparisons
- Time series trends
- Industry statistics
- Social indicators (education, healthcare, housing)

AI-based Generation:
===================
Characteristics:
- Creative, natural language prompts
- Context-aware based on CZSU data description
- Uses GPT-4o-mini with temperature=1.0 for variety
- Automatic fallback to default prompts on errors
- Validates minimum prompt count (3+)

Generation guidelines:
- Prompts can be questions, statements, or commands
- Concise and user-friendly
- Cover diverse data aspects (economy, population, finance, etc.)
- Natural conversational style

Strategy Selection Logic:
========================
The main function uses this decision flow:
1. Check FOLLOWUP_PROMPTS_STRATEGY environment variable
2. If "ai": Use AI-based generation
3. If "template" or unset: Use template-based generation (default)
4. Invalid values default to template-based generation

Related Modules:
===============
- my_agent.utils.models: Azure OpenAI LLM factory functions
- api.utils.debug: Debug logging utilities
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import random
import time
from typing import List
import langsmith as ls

from langchain_core.prompts import ChatPromptTemplate

from my_agent.utils.models import get_azure_openai_chat_llm

# ==============================================================================
# CONDITIONAL IMPORTS
# ==============================================================================
# Import debug function with fallback for when api.utils.debug is not available
try:
    from api.utils.debug import print__main_debug
except ImportError:
    # Fallback debug function when api.utils.debug is not available
    def print__main_debug(msg: str) -> None:
        """Fallback debug function when api.utils.debug is not available."""
        debug_mode = os.environ.get("DEBUG", "0")
        if debug_mode == "1":
            print(f"[MAIN-DEBUG] {msg}")
            import sys

            sys.stdout.flush()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def generate_initial_followup_prompts() -> List[str]:
    """Generate initial follow-up prompt suggestions for new conversations.

    This is the main entry point that selects between template-based and AI-based
    generation strategies based on the FOLLOWUP_PROMPTS_STRATEGY environment variable.

    Environment Variable Configuration:
    - FOLLOWUP_PROMPTS_STRATEGY="template" (default): Use template-based generation
    - FOLLOWUP_PROMPTS_STRATEGY="ai": Use AI-based generation with GPT-4o-mini

    Returns:
        List[str]: List of 5 follow-up prompt suggestions

    Example:
        >>> # With FOLLOWUP_PROMPTS_STRATEGY="template" (or unset)
        >>> prompts = generate_initial_followup_prompts()
        >>> len(prompts)
        5
        >>> # Returns template-based prompts

        >>> # With FOLLOWUP_PROMPTS_STRATEGY="ai"
        >>> prompts = generate_initial_followup_prompts()
        >>> len(prompts)
        5
        >>> # Returns AI-generated prompts
    """
    strategy = os.environ.get("FOLLOWUP_PROMPTS_STRATEGY", "template").lower()
    print__main_debug(
        f"🎯 PROMPT GEN: Using '{strategy}' strategy for followup prompts"
    )

    if strategy == "ai":
        print__main_debug("🤖 PROMPT GEN: Selected AI-based generation")
        return generate_initial_followup_prompts_ai()
    else:
        if strategy != "template":
            print__main_debug(
                f"⚠️ PROMPT GEN: Unknown strategy '{strategy}', defaulting to 'template'"
            )
        print__main_debug("📋 PROMPT GEN: Selected template-based generation")
        return generate_initial_followup_prompts_template()


# ==============================================================================
# TEMPLATE-BASED GENERATION
# ==============================================================================
def generate_initial_followup_prompts_template() -> List[str]:
    """Generate initial follow-up prompt suggestions using template-based approach.

    This function creates diverse starter suggestions for users beginning new chat sessions.
    It uses dynamic template-based generation with pseudo-random selections to ensure variety
    across different conversation starts. The generated prompts help users discover the types
    of questions they can ask about Czech Statistical Office data.

    The generation process:
    1. Uses current timestamp as seed for pseudo-randomness
    2. Selects from 15+ diverse prompt templates
    3. Fills templates with random selections from topic pools
    4. Ensures 5 unique prompts (no duplicate templates)
    5. Returns prompts suitable for CZSU data exploration

    Template categories:
    - Population and demographic queries
    - Economic indicators (GDP, wages, employment)
    - Regional comparisons
    - Time series trends
    - Industry statistics
    - Social indicators (education, healthcare, housing)

    Returns:
        List[str]: List of 5 dynamically generated follow-up prompt suggestions

    Example output:
        [
            "What are the population trends in Prague?",
            "Show me employment statistics by industry.",
            "Compare GDP growth across different years.",
            "What are the unemployment rates in major cities?",
            "Tell me about healthcare spending in Czech Republic."
        ]

    Note:
        This function is called for new conversations when template-based strategy is selected.
        Uses timestamp-based seeding for variety while maintaining determinism within milliseconds.
    """
    print__main_debug(
        "🎯 PROMPT GEN: Starting dynamic template-based prompt generation"
    )

    # ===========================================================================
    # STEP 1: INITIALIZE PSEUDO-RANDOM SEED FOR VARIETY
    # ===========================================================================
    # Use timestamp-based seed to ensure different prompts on each conversation start
    # while maintaining determinism within the same millisecond (useful for debugging)
    seed = int(time.time() * 1000) % 1000000  # Millisecond timestamp modulo 1M
    random.seed(seed)

    # ===========================================================================
    # STEP 2: DEFINE PROMPT TEMPLATES
    # ===========================================================================
    # Comprehensive set of prompt templates covering various CZSU data categories
    # Templates use placeholders like {region}, {metric}, {topic} filled in Step 4
    prompt_templates = [
        "What are the population trends in {region}?",
        "Show me employment statistics by {category}.",
        "Compare {metric} growth across different years.",
        "What are the latest statistics on {topic}?",
        "How has {indicator} changed in recent {period}?",
        "What are the {type} rates in {location}?",
        "Show me data about {subject} from {source}.",
        "What trends can you see in {area} statistics?",
        "Compare {metric} between {group1} and {group2}.",
        "What are the current {indicator} figures for {region}?",
        "Tell me about {topic} in {location}.",
        "Show me {subject} statistics for {period}.",
        "What is the {indicator} situation in {region}?",
        "Compare {metric} across {group1} and {group2}.",
        "What are the trends in {area} data?",
    ]

    # ===========================================================================
    # STEP 3: DEFINE PLACEHOLDER VALUE POOLS
    # ===========================================================================
    # Diverse value pools for filling template placeholders
    # Values represent common CZSU data categories and entities
    regions = [
        "Prague",
        "Czech Republic",
        "major cities",
        "different regions",
        "Brno",
    ]
    categories = [
        "region",
        "industry",
        "age group",
        "education level",
        "sector",
    ]
    metrics = [
        "GDP",
        "employment",
        "population",
        "export",
        "import",
        "wage",
    ]
    topics = [
        "crime rates",
        "healthcare spending",
        "education levels",
        "housing prices",
        "migration",
        "birth rates",
    ]
    periods = [
        "years",
        "quarters",
        "months",
        "decades",
        "recent years",
    ]
    types = [
        "unemployment",
        "inflation",
        "birth",
        "migration",
        "divorce",
    ]
    locations = [
        "Prague",
        "Brno",
        "Czech Republic",
        "major regions",
    ]
    subjects = [
        "agricultural production",
        "industrial output",
        "tourism numbers",
        "energy consumption",
        "trade balance",
    ]
    sources = [
        "government reports",
        "statistical surveys",
        "economic indicators",
        "census data",
        "official statistics",
    ]
    areas = [
        "labor market",
        "demographic",
        "economic",
        "environmental",
        "social",
        "health",
    ]
    indicators = [
        "unemployment",
        "inflation",
        "GDP growth",
        "population",
        "wage growth",
        "export growth",
    ]
    # Tuple pairs for comparison-style prompts (group1, group2)
    group1_group2 = [
        ("urban and rural areas", "rural areas"),
        ("men and women", "women"),
        ("young and old", "older population"),
        ("public and private sector", "private companies"),
        ("domestic and foreign", "foreign companies"),
        ("large and small enterprises", "small businesses"),
    ]

    # ===========================================================================
    # STEP 4: GENERATE 5 UNIQUE PROMPTS
    # ===========================================================================
    # Select templates randomly and fill placeholders with random values from pools
    generated_prompts = []
    used_templates = set()  # Track used templates to ensure uniqueness

    while len(generated_prompts) < 5:
        # Select random template
        template = random.choice(prompt_templates)

        # Skip if template already used (ensures 5 different templates)
        if template in used_templates:
            continue
        used_templates.add(template)

        # Fill in template placeholders with random selections from appropriate pools
        prompt = template
        if "{region}" in prompt:
            prompt = prompt.replace("{region}", random.choice(regions))
        if "{category}" in prompt:
            prompt = prompt.replace("{category}", random.choice(categories))
        if "{metric}" in prompt:
            prompt = prompt.replace("{metric}", random.choice(metrics))
        if "{topic}" in prompt:
            prompt = prompt.replace("{topic}", random.choice(topics))
        if "{period}" in prompt:
            prompt = prompt.replace("{period}", random.choice(periods))
        if "{type}" in prompt:
            prompt = prompt.replace("{type}", random.choice(types))
        if "{location}" in prompt:
            prompt = prompt.replace("{location}", random.choice(locations))
        if "{subject}" in prompt:
            prompt = prompt.replace("{subject}", random.choice(subjects))
        if "{source}" in prompt:
            prompt = prompt.replace("{source}", random.choice(sources))
        if "{area}" in prompt:
            prompt = prompt.replace("{area}", random.choice(areas))
        if "{indicator}" in prompt:
            prompt = prompt.replace("{indicator}", random.choice(indicators))
        if "{group1} and {group2}" in prompt:
            g1, g2 = random.choice(group1_group2)
            prompt = prompt.replace("{group1}", g1).replace("{group2}", g2)

        generated_prompts.append(prompt)

    # ===========================================================================
    # STEP 5: LOG AND RETURN RESULTS
    # ===========================================================================
    final_prompts = generated_prompts
    print__main_debug(f"🎲 Generated {len(final_prompts)} dynamic prompts")
    for i, p in enumerate(final_prompts, 1):
        print__main_debug(f"   {i}. {p}")

    return final_prompts


# ==============================================================================
# AI-BASED GENERATION
# ==============================================================================
def generate_initial_followup_prompts_ai() -> List[str]:
    """Generate initial follow-up prompt suggestions using AI.

    This function uses Azure OpenAI GPT-4o-mini to generate creative, context-aware
    starter suggestions before the LangGraph workflow executes. These prompts will be
    displayed to users when they start a new chat, giving them ideas for questions
    they can ask about Czech Statistical Office data.

    The AI generation process:
    1. Initialize GPT-4o-mini with temperature=1.0 for creativity
    2. Provide system prompt with CZSU data context and guidelines
    3. Request 5 diverse prompts covering different data aspects
    4. Parse and validate LLM response
    5. Fallback to default prompts if generation fails or insufficient prompts

    Generation guidelines (instructed to LLM):
    - Prompts don't have to be questions - can be statements, commands, or intents
    - Be concise and to the point - brief prompts preferred
    - Cover different aspects: economy, population, finance, etc.
    - Natural and user-friendly language
    - One prompt per line, no numbering

    Returns:
        List[str]: A list of AI-generated suggested follow-up prompts (maximum 5)

    Raises:
        No exceptions raised - errors are caught and trigger fallback to default prompts

    Example output:
        [
            "Show me the latest GDP trends",
            "What are employment rates by region?",
            "Compare inflation across recent years",
            "Tell me about population changes in Prague",
            "How has healthcare spending evolved?"
        ]

    Fallback behavior:
        If AI generation fails (API error, parsing error, etc.) or returns fewer than 3 prompts,
        the function automatically falls back to these default prompts:
        - "What are the population trends in Prague?"
        - "Show me employment statistics by region"
        - "Compare GDP growth across different years"

    Note:
        This function is called for new conversations when AI-based strategy is selected
        via FOLLOWUP_PROMPTS_STRATEGY="ai" environment variable.
    """
    print__main_debug("🎯 PROMPT GEN: Starting AI-based initial prompt generation")
    try:
        # Use the same model as other nodes but with temperature 1.0 for creativity
        llm = get_azure_openai_chat_llm(
            deployment_name="gpt-4o-mini-mimi2",
            model_name="gpt-4o-mini",
            openai_api_version="2024-05-01-preview",
            temperature=0.0,
        )

        # llm = get_azure_openai_chat_llm(
        #     deployment_name="gpt-5-nano_mimi_test",
        #     model_name="gpt-5-nano",
        #     openai_api_version="2024-12-01-preview",
        #     temperature=1.0,
        # )
        print__main_debug("🤖 PROMPT GEN: LLM initialized with temperature=1.0")

        # Define a large pool of topics to ensure diversity
        all_topics = [
            "population trends",
            "GDP growth",
            "employment rates",
            "inflation data",
            "trade balance",
            "environmental emissions",
            "climate data",
            "national parks",
            "waste generation",
            "water supply",
            "birth rates",
            "death causes",
            "migration patterns",
            "foreign investment",
            "government debt",
            "tax receipts",
            "capital formation",
            "monetary aggregates",
            "exchange rates",
            "consumer prices",
            "household income",
            "labour costs",
            "unemployment rates",
            "business births",
            "export volumes",
            "import values",
            "real estate prices",
            "agricultural prices",
            "energy consumption",
            "housing costs",
        ]

        # Randomly select 10 topics to force variety in the prompt context
        # This ensures that even with the same system prompt structure, the context changes
        selected_topics = random.sample(all_topics, min(10, len(all_topics)))
        topics_str = ", ".join(selected_topics)

        system_prompt = f"""
You are a prompt generation assistant for a Czech Statistical Office data analysis system.

About our data:
Summary data on Czechia provides selected data from individual areas with a focus on the real economy, 
monetary and fiscal indicators. They gather data from the CZSO as well as data from other institutions, 
such as the Czech National Bank, the Ministry of Finance and others.

Your task: Generate exactly 5 diverse prompts that users might use to get answers or information from this data.

Important guidelines:
- Prompts don't have to be questions - they can be statements, commands, or other types of intents
- Be concise and to the point - prompts should be brief
- Each prompt should be on a new line
- Don't number the prompts
- Cover different aspects of the available data (economy, population, finance, etc.)
- Make them natural and user-friendly

Here are some RANDOM example topics to draw inspiration from (use these to ensure variety): {topics_str}.
"""

        human_prompt = "Generate 5 prompts for users to explore Czech statistical data."

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )
        print__main_debug("📝 PROMPT GEN: Prompt template created, invoking LLM")

        # Synchronous invoke since this is called from synchronous context
        result = llm.invoke(prompt.format_messages())
        generated_text = result.content.strip()
        print__main_debug(
            f"✅ PROMPT GEN: LLM returned {len(generated_text)} characters"
        )

        # Parse the generated prompts (split by newlines and filter empty lines)
        prompts = [line.strip() for line in generated_text.split("\n") if line.strip()]
        print__main_debug(
            f"📋 PROMPT GEN: Parsed {len(prompts)} prompts from LLM response"
        )

        # Ensure we have at least some prompts, fallback if needed
        if len(prompts) < 3:
            print__main_debug(
                f"⚠️ PROMPT GEN: Only {len(prompts)} prompts generated, using fallback"
            )
            prompts = [
                "What are the population trends in Prague?",
                "Show me employment statistics by region",
                "Compare GDP growth across different years",
            ]

        final_prompts = prompts[:5]  # Return maximum 5 prompts
        print__main_debug(f"💡 PROMPT GEN: Returning {len(final_prompts)} prompts")
        for i, p in enumerate(final_prompts, 1):
            print__main_debug(f"   {i}. {p}")

        return final_prompts

    except Exception as e:
        # Fallback to default prompts if AI generation fails
        print__main_debug(f"❌ PROMPT GEN: Failed to generate AI prompts - {str(e)}")
        print__main_debug("🔄 PROMPT GEN: Using fallback default prompts")
        return [
            "What are the population trends in Prague?",
            "Show me employment statistics by region",
            "Compare GDP growth across different years",
        ]
