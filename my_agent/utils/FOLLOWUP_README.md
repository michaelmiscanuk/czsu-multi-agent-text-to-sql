# Follow-up Prompt Generation Module

This module (`my_agent/utils/followup.py`) provides functionality for generating follow-up prompt suggestions that help users explore Czech Statistical Office (CZSU) data.

## Overview

The module offers two generation strategies:

1. **Template-based generation**: Fast, deterministic prompts using predefined templates
2. **AI-based generation**: Creative, context-aware prompts using Azure OpenAI GPT-4o-mini

The generation strategy is selected via the `FOLLOWUP_PROMPTS_STRATEGY` environment variable.

## Configuration

Add to your `.env` file:

```bash
# Strategy for generating initial followup prompts for new conversations
# Options: "template" (default, fast, no API calls) or "ai" (uses GPT-4o-mini, more creative)
FOLLOWUP_PROMPTS_STRATEGY=template
```

### Strategy Options

- **`template`** (default):
  - Fast execution (no API calls)
  - Deterministic with pseudo-random variety
  - 15+ diverse prompt templates
  - Multiple topic pools for placeholder substitution
  - Ensures 5 unique prompts per call

- **`ai`**:
  - Creative, natural language prompts
  - Context-aware based on CZSU data description
  - Uses GPT-4o-mini with temperature=1.0 for variety
  - Automatic fallback to default prompts on errors
  - Requires Azure OpenAI API credentials

## Functions

### `generate_initial_followup_prompts()`

Main entry point that selects strategy based on `FOLLOWUP_PROMPTS_STRATEGY` env variable.

```python
from my_agent.utils.followup import generate_initial_followup_prompts

prompts = generate_initial_followup_prompts()
# Returns 5 prompts using configured strategy
```

### `generate_initial_followup_prompts_template()`

Explicitly uses template-based generation regardless of env variable.

```python
from my_agent.utils.followup import generate_initial_followup_prompts_template

prompts = generate_initial_followup_prompts_template()
# Always returns template-based prompts
```

### `generate_initial_followup_prompts_ai()`

Explicitly uses AI-based generation regardless of env variable.

```python
from my_agent.utils.followup import generate_initial_followup_prompts_ai

prompts = generate_initial_followup_prompts_ai()
# Always returns AI-generated prompts
```

## Usage Examples

### Basic Usage (Respects Configuration)

```python
from my_agent.utils.followup import generate_initial_followup_prompts

# Uses strategy from FOLLOWUP_PROMPTS_STRATEGY env var
prompts = generate_initial_followup_prompts()

for i, prompt in enumerate(prompts, 1):
    print(f"{i}. {prompt}")
```

### Force Template-Based Generation

```python
from my_agent.utils.followup import generate_initial_followup_prompts_template

# Always uses template-based generation
prompts = generate_initial_followup_prompts_template()
```

### Force AI-Based Generation

```python
from my_agent.utils.followup import generate_initial_followup_prompts_ai

# Always uses AI-based generation
# Requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env
prompts = generate_initial_followup_prompts_ai()
```

## Template-Based Generation Details

### Template Categories

- Population and demographic queries
- Economic indicators (GDP, wages, employment)
- Regional comparisons
- Time series trends
- Industry statistics
- Social indicators (education, healthcare, housing)

### Example Templates

```python
"What are the population trends in {region}?"
"Show me employment statistics by {category}."
"Compare {metric} growth across different years."
"How has {indicator} changed in recent {period}?"
```

### Placeholder Pools

- **regions**: Prague, Czech Republic, major cities, different regions, Brno
- **categories**: region, industry, age group, education level, sector
- **metrics**: GDP, employment, population, export, import, wage
- **topics**: crime rates, healthcare spending, education levels, housing prices, migration, birth rates
- **periods**: years, quarters, months, decades, recent years
- And more...

### Example Output

```
1. Show me trade balance statistics for decades.
2. Compare export growth across different years.
3. What are the latest statistics on healthcare spending?
4. Show me employment statistics by region.
5. What are the population trends in major cities?
```

## AI-Based Generation Details

### System Prompt

The AI is instructed to:
- Generate exactly 5 diverse prompts
- Cover different aspects: economy, population, finance, etc.
- Use natural, user-friendly language
- Make prompts concise and to the point
- Accept questions, statements, or commands

### Example Output

```
1. Show me the latest GDP trends
2. What are employment rates by region?
3. Compare inflation across recent years
4. Tell me about population changes in Prague
5. How has healthcare spending evolved?
```

### Fallback Behavior

If AI generation fails or returns insufficient prompts (< 3), the function automatically falls back to:

```python
[
    "What are the population trends in Prague?",
    "Show me employment statistics by region",
    "Compare GDP growth across different years",
]
```

## Testing

### Standalone Test (Recommended)
Due to import dependencies, use the standalone test script:

```bash
python tests/other/test_followup.py
```

This test includes:
1. Strategy selection logic testing
2. Template-based generation testing
3. AI-based generation testing (commented out to avoid API calls)

### Direct Module Testing
For testing within the full application context:

```bash
# From project root
python -c "from my_agent.utils.followup import generate_initial_followup_prompts_template; prompts = generate_initial_followup_prompts_template(); print(f'Generated {len(prompts)} prompts')"
```

### Integration Testing
The module is automatically tested when running the main application:

```bash
python main.py "test query"
```

This will use the configured strategy (`FOLLOWUP_PROMPTS_STRATEGY`) to generate initial follow-up prompts for new conversations.

## Performance Considerations

### Template-Based Strategy
- **Speed**: ~1-2ms per call
- **API Calls**: None
- **Cost**: Free
- **Variability**: Pseudo-random with timestamp seed

### AI-Based Strategy
- **Speed**: ~500-2000ms per call (depends on API latency)
- **API Calls**: 1 per call
- **Cost**: ~$0.0001 per call (GPT-4o-mini pricing)
- **Variability**: High (temperature=1.0)

## Recommendations

### When to Use Template-Based Strategy
- Production environments where speed matters
- Cost-sensitive applications
- When predictable prompt patterns are acceptable
- When API availability is a concern

### When to Use AI-Based Strategy
- When maximum creativity and variety is needed
- When natural language flow is critical
- When willing to trade speed/cost for quality
- For demos or special user experiences

## Migration Notes

This module was extracted from `main.py` to:
1. Improve code organization and maintainability
2. Enable easy switching between strategies
3. Allow for future expansion (e.g., new strategies)
4. Facilitate testing and debugging

The original `generate_initial_followup_prompts()` function has been:
- Moved to `my_agent/utils/followup.py`
- Renamed to `generate_initial_followup_prompts_template()`
- Wrapped by new main function that handles strategy selection
- Extended with AI-based alternative