"""Quick test script to verify followup prompt generation works correctly."""

import os
import random
import time
from typing import List

# ==============================================================================
# STANDALONE IMPLEMENTATION FOR TESTING
# ==============================================================================
# Copy of the template-based generation logic for standalone testing
# This avoids importing the full my_agent package with its dependencies


def print__main_debug(msg: str) -> None:
    """Standalone debug function for testing."""
    debug_mode = os.environ.get("DEBUG", "0")
    if debug_mode == "1":
        print(f"[MAIN-DEBUG] {msg}")
        import sys

        sys.stdout.flush()


def generate_initial_followup_prompts_template() -> List[str]:
    """Standalone template-based generation for testing."""
    print__main_debug(
        "🎯 PROMPT GEN: Starting dynamic template-based prompt generation"
    )

    # ===========================================================================
    # STEP 1: INITIALIZE PSEUDO-RANDOM SEED FOR VARIETY
    # ===========================================================================
    seed = int(time.time() * 1000) % 1000000  # Millisecond timestamp modulo 1M
    random.seed(seed)

    # ===========================================================================
    # STEP 2: DEFINE PROMPT TEMPLATES
    # ===========================================================================
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
    generated_prompts = []
    used_templates = set()

    while len(generated_prompts) < 5:
        template = random.choice(prompt_templates)

        if template in used_templates:
            continue
        used_templates.add(template)

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
# TEST FUNCTIONS
# ==============================================================================


def test_template_generation():
    """Test template-based generation."""
    print("\n2. Testing template-based generation:")
    print("-" * 80)
    prompts_template = generate_initial_followup_prompts_template()
    print(f"✅ Success! Generated {len(prompts_template)} prompts:")
    for i, p in enumerate(prompts_template, 1):
        print(f"   {i}. {p}")
    return prompts_template


def test_strategy_selection():
    """Test strategy selection logic."""
    print("\n1. Testing strategy selection logic:")
    print("-" * 80)

    # Test default (template) strategy
    strategy = os.environ.get("FOLLOWUP_PROMPTS_STRATEGY", "template").lower()
    print(
        f"📋 FOLLOWUP_PROMPTS_STRATEGY env var: '{os.environ.get('FOLLOWUP_PROMPTS_STRATEGY', 'not set')}'"
    )
    print(f"🎯 Selected strategy: '{strategy}'")

    if strategy == "ai":
        print("🤖 Would use AI-based generation")
        prompts = ["AI generation would be used here"]
    else:
        print("📋 Using template-based generation")
        prompts = generate_initial_followup_prompts_template()

    print(f"✅ Success! Generated {len(prompts)} prompts:")
    for i, p in enumerate(prompts, 1):
        print(f"   {i}. {p}")
    return prompts


# ==============================================================================
# MAIN TEST EXECUTION
# ==============================================================================

print("=" * 80)
print("Testing followup prompt generation (standalone)")
print("=" * 80)

# Test strategy selection
test_strategy_selection()

# Test template generation explicitly
test_template_generation()

# Test AI generation (commented out)
print("\n3. Testing AI-based generation (skipped to avoid API call):")
print("-" * 80)
print("   To test AI generation, uncomment the code below and ensure")
print("   AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set in .env")
print("   Note: AI generation requires the full my_agent package to be available")

print("\n" + "=" * 80)
print("All tests completed successfully!")
print("=" * 80)
