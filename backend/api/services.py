import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List

import google.generativeai as genai
import spacy
from tavily import AsyncTavilyClient
from django.conf import settings

from .db_config import conversations_collection

_phase_timers = {}
logger = logging.getLogger(__name__)


class ProviderError(Exception):
    def __init__(self, provider: str, public_message: str):
        super().__init__(public_message)
        self.provider = provider
        self.public_message = public_message


async def with_timeout(coro, timeout_seconds: int, provider: str, public_message: str):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        logger.warning("%s timed out after %ss.", provider, timeout_seconds)
        raise ProviderError(provider, public_message) from exc


async def iter_with_timeout(async_iterable, timeout_seconds: int, provider: str, public_message: str):
    iterator = async_iterable.__aiter__()
    while True:
        try:
            item = await asyncio.wait_for(iterator.__anext__(), timeout=timeout_seconds)
            yield item
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError as exc:
            logger.warning("%s stream timed out after %ss.", provider, timeout_seconds)
            raise ProviderError(provider, public_message) from exc


def phase_start(name: str):
    _phase_timers[name] = time.perf_counter()
    logger.info("[TIMER] %s started", name)


def phase_end(name: str):
    start = _phase_timers.pop(name, None)
    if start:
        elapsed = time.perf_counter() - start
        logger.info("[TIMER] %s finished in %.2fs", name, elapsed)
        return elapsed
    return 0.0


try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    logger.info("Google Generative AI configured.")
except Exception as e:
    logger.exception("Failed to configure Google Generative AI: %s", e)


def _format_context_for_prompt(context_package: Dict[str, Any]) -> str:
    """Formats the conversation history into a clean, readable string for LLM prompts."""
    history_str = "No previous conversation history."
    
    if context_package and context_package.get('previous_turns'):
        formatted_turns = []
        for i, turn in enumerate(context_package['previous_turns'], 1):
            formatted_turns.append(
                f"Turn {i}:\n"
                f"  - User Query: \"{turn.get('query')}\"\n"
                f"  - Assistant's Summary: \"{turn.get('summary')}\""
            )
        if formatted_turns:
            history_str = "\n".join(formatted_turns)
            
    return history_str


def _clean_snippet(text: str, max_length: int = 420) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length].rsplit(" ", 1)[0] + "..."


def _build_snippet_fallback_answer(
    prompt: str,
    scraped_data: List[Dict[str, str]],
    instant_answer: str | None = None,
) -> str:
    """Builds a readable cited answer when the LLM synthesis provider is slow."""
    lines = []

    if instant_answer:
        lines.append(f"> **Quick Summary:** {instant_answer}")
        lines.append("")

    lines.append("## Source-backed fallback")
    lines.append("")
    lines.append(
        "The synthesis model took too long, so ARGON is showing a compact answer from the retrieved sources instead."
    )
    lines.append("")

    usable_items = [
        item for item in scraped_data
        if _clean_snippet(item.get("content", ""))
    ][:4]

    if not usable_items:
        lines.append("The search completed, but the returned snippets did not contain enough readable text to summarize safely.")
        return "\n".join(lines)

    lines.append(f"**Query:** {prompt}")
    lines.append("")
    lines.append("### Key points")
    for index, item in enumerate(usable_items, 1):
        title = item.get("title") or item.get("source") or f"Source {index}"
        snippet = _clean_snippet(item.get("content", ""))
        lines.append(f"- **{title}**: {snippet} [{index}]")

    lines.append("")
    lines.append("### Sources")
    for index, item in enumerate(usable_items, 1):
        title = item.get("title") or f"Source {index}"
        source = item.get("source") or ""
        lines.append(f"{index}. [{title}]({source})" if source else f"{index}. {title}")

    return "\n".join(lines)


def extract_contextual_metadata(prompt: str) -> Dict[str, bool]:
    """
    Performs a rapid, computationally cheap analysis of the query's form
    to extract key contextual metadata before deeper processing.
    """
    print("[CLASSIFIER STAGE 1] Extracting contextual metadata...")
    
    prompt_lower = prompt.lower()
    
    # 1. Content Analysis
    demonstrative_markers = [
        "this code", "my code", "this text", "my text", "this document", 
        "my document", "the following", "above code", "below code",
        "this error", "my error", "this function", "my function",
        "this script", "my script", "attached file", "my essay",
        "this paragraph", "my paragraph", "these lines", "this snippet"
    ]
    
    has_attached_content = (
        any(marker in prompt_lower for marker in demonstrative_markers) or
        bool(re.search(r'```[\s\S]*```', prompt)) or  # Convert to bool
        bool(re.search(r'\{[\s\S]{20,}\}', prompt)) or
        bool(re.search(r'\bthis\s+(code|text|error|function|bug|issue|problem)\b', prompt_lower)) or
        bool(re.search(r'\bmy\s+(code|text|error|function|project|assignment)\b', prompt_lower))
    )
    
    # 2. Temporal Analysis
    temporal_keywords = [
        "latest", "current", "today", "recent", "now", "breaking",
        "just announced", "this week", "this month", "this year",
        "update", "news", "right now", "at the moment", "presently"
    ]
    
    current_year = str(datetime.now().year)
    previous_year = str(datetime.now().year - 1)
    
    is_temporal = (
        any(keyword in prompt_lower for keyword in temporal_keywords) or
        current_year in prompt or
        previous_year in prompt or
        bool(re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}\b', prompt_lower)) or
        bool(re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', prompt))
    )
    
    # 3. Volatile Domain Detection
    volatile_domains = [
        "news", "stock", "stocks", "market", "weather", "election",
        "breaking", "score", "game", "match", "price", "cryptocurrency",
        "crypto", "bitcoin", "trending", "viral", "poll"
    ]
    
    is_volatile_domain = any(domain in prompt_lower for domain in volatile_domains)
    
    # 4. Generation Intent Detection
    generation_verbs = [
        "create", "write", "generate", "make", "build", "compose",
        "draft", "design", "develop", "code", "implement", "construct",
        "craft", "produce"
    ]
    
    creative_content_types = [
        "story", "poem", "song", "lyrics", "essay", "email", "letter",
        "script", "joke", "recipe", "paragraph", "article", "blog"
    ]
    
    computational_indicators = [
        "solve", "calculate", "compute", "find the value", "evaluate",
        "simplify", "factor", "integrate", "differentiate"
    ]
    
    is_generation_task = (
        any(verb in prompt_lower for verb in generation_verbs) and
        any(content in prompt_lower for content in creative_content_types)
    ) or any(indicator in prompt_lower for indicator in computational_indicators)
    
    # 5. Factual Lookup Patterns
    factual_patterns = [
        "who is", "who are", "what is", "what are", "where is", "when did",
        "when was", "how does", "how did", "why does", "why did",
        "explain", "describe", "tell me about", "what happened",
        "give me information", "information about"
    ]
    
    is_factual_lookup = any(pattern in prompt_lower for pattern in factual_patterns)
    
    # 6. Entity Indicators
    entity_indicators = [
        " ceo ", " founder ", " president ", " company ", " organization ",
        " university ", " country ", " city ", " person ", " celebrity ",
        " scientist ", " author ", " politician "
    ]
    
    likely_has_entities = any(indicator in f" {prompt_lower} " for indicator in entity_indicators)
    
    metadata = {
        'has_attached_content': has_attached_content,
        'is_temporal': is_temporal,
        'is_volatile_domain': is_volatile_domain,
        'is_generation_task': is_generation_task,
        'is_factual_lookup': is_factual_lookup,
        'likely_has_entities': likely_has_entities
    }
    
    print(f"  > Metadata extracted: {metadata}")
    return metadata

# NLP implementation 

try:
    nlp = spacy.load("en_core_web_sm")
    print("[CLASSIFIER STAGE 2] spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("[CLASSIFIER STAGE 2] spaCy model not found. Please run 'python -m spacy download en_core_web_sm'")
    nlp = None

# Defining the programmatic mapping from LLM classifications to numerical scores
SCORE_MAPPING = {
    'intent_type': {
        'factual_explanation': 0.9,
        'general_qa': 0.7,
        'comparison': 0.75,
        'analytical_reasoning': 0.4,
        'code_generation': 0.2,
        'creative_generation': 0.1,
        'math_computation': 0.15,
        'content_summarization': 0.3
    },
    'entity_type': {
        'specific_person_or_event': 0.9, # dynamism
        'organization_or_product': 0.8, # dynamism
        'broad_concept': 0.2, # dynamism
        'user_provided_content': 0.1, # dynamism
        'abstract_idea': 0.1, # dynamism
    },
    'information_scope': {
        'comprehensive_overview': 0.9, # comprehensiveness
        'specific_answer': 0.2, # comprehensiveness
        'step_by_step_guide': 0.7 # comprehensiveness
    },
    'verification_level': {
        'high_verification': 0.9, # verification
        'medium_verification': 0.5, # verification
        'low_verification': 0.1, # verification
    }
}

def _get_linguistic_features(prompt: str) -> Dict[str, Any]:
    """Helper to perform deterministic NLP processing using spaCy."""
    if not nlp:
        return {}
    
    doc = nlp(prompt)
    
    entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
    
    # Simple verb extraction
    root_verb = "unknown"
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            root_verb = token.lemma_
            break
            
    return {
        "entities": entities,
        "root_verb": root_verb,
        "is_question": prompt.strip().endswith('?')
    }

def _get_llm_classifications(prompt: str, linguistic_features: Dict[str, Any], context_metadata: Dict[str, Any], context_package: Dict[str, Any]) -> Dict[str, str]:
    """Gets semantic routing labels from Gemini."""
    
    # Use our helper to format the conversation history
    conversation_history = _format_context_for_prompt(context_package)

    system_prompt = f"""
### ROLE ###
You are a highly-tuned NLP classification model. Your purpose is to act as the semantic reasoning core of a sophisticated query processing pipeline. You will receive a user's query, pre-processed metadata, linguistic features, and the conversation history. Your sole task is to analyze all this information and classify the query's core attributes.

### INPUTS ###
You will be given four pieces of information:

1.  **Contextual Metadata (from fast, rule-based checks):**
    ```json
    {json.dumps(context_metadata, indent=4)}
    ```

2.  **Linguistic Features (from spaCy NLP analysis):**
    ```json
    {json.dumps(linguistic_features, indent=4)}
    ```

3.  **Conversation History:**
    {conversation_history}

4.  **Raw User Query:**
    `{prompt}`

### REASONING FRAMEWORK ###
1.  **Analyze Relationship First:** Your primary task is to determine the relationship between the `Raw User Query` and the `Conversation History`.
    *   Is it a direct follow-up (e.g., "what about its applications?")?
    *   Is it a completely new, unrelated topic?
    *   This relationship is the most important signal for your classifications.

2.  **Analyze Context & Linguistics:** Now consider the other inputs.
    *   If `has_attached_content` is `true`, `Entity Type` is likely `user_provided_content`.
    *   If `is_temporal` is `true`, `Verification Level` should be `high_verification`.
    *   The `root_verb` (e.g., "explain", "create", "compare") is a strong clue for `Intent Type`.

### OUTPUT CONSTRAINTS ###
You MUST respond with ONLY a valid JSON object with four keys.

-   **`intent_type`**: (Choose one: `factual_explanation`, `general_qa`, `comparison`, `analytical_reasoning`, `code_generation`, `creative_generation`, `math_computation`, `content_summarization`)
-   **`entity_type`**: (Choose one: `specific_person_or_event`, `organization_or_product`, `broad_concept`, `user_provided_content`, `abstract_idea`)
-   **`information_scope`**: (Choose one: `comprehensive_overview`, `specific_answer`, `step_by_step_guide`)
-   **`verification_level`**: (Choose one: `high_verification`, `medium_verification`, `low_verification`)
"""
    
    try:
        model = genai.GenerativeModel(
            model_name=settings.GEMINI_CLASSIFIER_MODEL,
            generation_config={"response_mime_type": "application/json"}
        )
        response = model.generate_content(system_prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"[CLASSIFIER STAGE 2] Error during LLM classification: {e}")
        return {
            "intent_type": "general_qa",
            "entity_type": "broad_concept",
            "information_scope": "specific_answer",
            "verification_level": "medium_verification",
        }


def generate_nlp_features_and_scores(prompt: str, context_metadata: Dict[str, bool], context_package: Dict[str, Any]) -> Dict[str, float]:
    """
    The main function for Stage 2. Orchestrates NLP processing, LLM classification,
    and programmatic scoring.
    
    Args:
        prompt: The raw user query.
        context_metadata: The output from Stage 1.

    Returns:
        A dictionary of the final six numerical scores.
    """
    print("[CLASSIFIER STAGE 2] Generating NLP features and scores...")
    
    linguistic_features = _get_linguistic_features(prompt)
    print(f"  > Linguistic Features (spaCy): {linguistic_features}")
    
    llm_classifications = _get_llm_classifications(prompt, linguistic_features, context_metadata, context_package)

    print(f"  > LLM Classifications: {llm_classifications}")

    scores = {}
    scores['intent_type_score'] = SCORE_MAPPING['intent_type'].get(llm_classifications.get('intent_type'), 0.5)
    scores['entity_dynamism_score'] = SCORE_MAPPING['entity_type'].get(llm_classifications.get('entity_type'), 0.5)
    scores['comprehensiveness_score'] = SCORE_MAPPING['information_scope'].get(llm_classifications.get('information_scope'), 0.5)
    scores['verification_need_score'] = SCORE_MAPPING['verification_level'].get(llm_classifications.get('verification_level'), 0.5)

    if context_metadata['has_attached_content']:
        scores['context_dependency_score'] = 0.9
        scores['entity_dynamism_score'] = 0.1 
    else:
        scores['context_dependency_score'] = 0.1

    if context_metadata['is_temporal']:
        scores['temporal_urgency_score'] = 0.9
        scores['entity_dynamism_score'] = max(scores.get('entity_dynamism_score', 0.5), 0.8)
    else:
        scores['temporal_urgency_score'] = 0.1

    print(f"  > Final Scores: {scores}")
    return scores

# Final call 

CLASSIFIER_WEIGHTS = {
    'intent_type_score': 0.32,
    'entity_dynamism_score': 0.20,
    'temporal_urgency_score': 0.20,
    # This weight is NEGATIVE High context dependency strongly
    # penalizes the score, pushing it towards a direct answer.
    'context_dependency_score': -0.25, # Adjusted for stronger impact
    'verification_need_score': 0.18, # Slightly increased weight
    'comprehensiveness_score': 0.05,
}

# The threshold determines the cutoff for triggering a web search.
# A higher score indicates a stronger signal for needing web access.
DECISION_THRESHOLD = 0.50

def make_routing_decision(scores: Dict[str, float]) -> str:
    """
    Takes the final numerical scores and applies a weighted formula to make
    the definitive routing decision. This is a purely deterministic calculation.

    Args:
        scores: A dictionary of the six numerical scores from Stage 2.

    Returns:
        A string: 'search_required' or 'direct_answer'.
    """
    print("[CLASSIFIER STAGE 3] Making final routing decision...")

    # Ensure all expected scores are present, defaulting to a neutral 0.5 if not
    required_keys = CLASSIFIER_WEIGHTS.keys()
    for key in required_keys:
        if key not in scores:
            print(f"  > WARNING: Missing score for '{key}'. Defaulting to 0.5.")
            scores[key] = 0.5

    # Calculate the final weighted score
    decision_score = (
        scores['intent_type_score'] * CLASSIFIER_WEIGHTS['intent_type_score'] +
        scores['entity_dynamism_score'] * CLASSIFIER_WEIGHTS['entity_dynamism_score'] +
        scores['temporal_urgency_score'] * CLASSIFIER_WEIGHTS['temporal_urgency_score'] +
        scores['context_dependency_score'] * CLASSIFIER_WEIGHTS['context_dependency_score'] +
        scores['verification_need_score'] * CLASSIFIER_WEIGHTS['verification_need_score'] +
        scores['comprehensiveness_score'] * CLASSIFIER_WEIGHTS['comprehensiveness_score']
    )
    
    print(f"  > Calculated Decision Score: {decision_score:.4f}")
    print(f"  > Comparison Threshold: {DECISION_THRESHOLD}")

    if (
        scores["entity_dynamism_score"] <= 0.25
        and scores["temporal_urgency_score"] <= 0.2
        and scores["verification_need_score"] <= 0.5
    ):
        print("  > Stable concept detected. Using direct answer to avoid unnecessary provider fan-out.")
        return "direct_answer"

    # Apply the threshold to make the final decision
    if decision_score > DECISION_THRESHOLD:
        path = "search_required"
    else:
        path = "direct_answer"
        
    print(f"  > Final Path Decided: '{path}'")
    return path

async def get_intelligent_path(prompt: str, context_package: Dict[str, Any]) -> str:
    """Runs the routing pipeline and returns the answer path."""
    print("[CLASSIFIER] Starting routing pipeline.")

    phase_start("classifier") 

    context_metadata = extract_contextual_metadata(prompt)
    
    loop = asyncio.get_running_loop()
    scores = await with_timeout(
        loop.run_in_executor(None, generate_nlp_features_and_scores, prompt, context_metadata, context_package),
        settings.METADATA_TIMEOUT_SECONDS,
        "Gemini classifier",
        "The routing model took too long, so ARGON could not safely choose an answer path. Please try again.",
    )
    
    final_path = make_routing_decision(scores)

    phase_end("classifier") 

    print(f"[CLASSIFIER] Routing finished. Final path: {final_path}")
    
    return final_path



async def _search_tavily_async(query: str) -> Dict[str, Any]:
    """
    ULTRA-FAST TAVILY FETCH:
    - Uses 'advanced' depth for quality but 'raw_content=False' for speed.
    - Uses 'include_answer' to get Tavily's instant summary.
    """

    trigger_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[TAVILY] API Call Triggered at: {trigger_time}")


    tavily_api_key = os.getenv("TAVILY_API_KEY")
    tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

    try:
        phase_start("tavily_api_latency")

        response = await with_timeout(
            tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=6,
                include_answer=True,
                include_raw_content=False
            ),
            settings.TAVILY_TIMEOUT_SECONDS,
            "Tavily web search",
            "Live web search is taking too long. Try again, or turn off web search for a direct answer.",
        )

        phase_end("tavily_api_latency")

        return response
    except Exception as e:

        phase_end("tavily_api_latency")

        logger.warning("Tavily web search failed: %s", e)
        return {
            "results": [],
            "answer": None,
            "provider_error": "Live web search is temporarily unavailable. ARGON will continue with a direct answer.",
        }
    

async def _get_images_from_tavily_async(query: str) -> List[Dict[str, Any]]:
    """
    An asynchronous helper function to fetch ONLY image results for a given query
    using the Tavily Search API. This function is designed for resilience.

    Args:
        query: The search query string optimized for finding images.

    Returns:
        A list of image result dictionaries from Tavily, or an empty list if an error occurs.
    """
    try:

        print(f"  > Starting IMAGE search for: '{query}'")

        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            print("[TAVILY] TAVILY_API_KEY not found in environment variables.")
            return []
        
        tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
        
        # The core of this function make the API call with include_images=True
        response = await with_timeout(
            tavily_client.search(
                query=query,
                search_depth="basic",
                include_images=True,
                max_results=15
            ),
            settings.TAVILY_TIMEOUT_SECONDS,
            "Tavily image search",
            "Image search is temporarily unavailable.",
        )
        

        images = response.get("images", [])
        
        print(f"  < Finished IMAGE search for: '{query}'. Found {len(images)} images.")
        return images
        
    except Exception as e:
        logger.warning("Tavily image search failed for %s: %s", query, e)
        return []


async def generate_ui_spec_from_markdown(markdown_content: str, context_package: Dict[str, Any]) -> str:
    """Converts the final Markdown answer into a Thesys C1 response."""
    print("[THESYS] Converting Markdown to C1.")

    phase_start("thesys")
    
    conversation_history = _format_context_for_prompt(context_package)

    thesys_meta_prompt = f"""
You are a world-class UI/UX architect specializing in transforming text into intuitive, visually engaging interfaces. Your goal is to create a UI that maximizes comprehension, engagement, and usability.

=== CORE DESIGN PRINCIPLES ===
1. **Clarity First**: The UI must make the information EASIER to understand than plain text
2. **Progressive Disclosure**: Show essential info first, hide complexity behind interactions
3. **Visual Hierarchy**: Use size, color, and spacing to guide the eye
4. **Scannable**: Users should grasp the structure in 2-3 seconds
5. **Content Distillation**: The raw input may contain boilerplate or fragments. You MUST intelligently filter out irrelevant snippets and only render the core, high-value information into the UI components.

=== CONVERSATION CONTEXT ===
{conversation_history}

=== CONTENT TYPE DETECTION & STRATEGY ===
Analyze the markdown content and determine its primary type, then apply the appropriate UI strategy:

**IF EXPLANATORY/EDUCATIONAL** (how-to, concepts, definitions):
- Use progressive reveal sections with "expand to learn more"
- Add visual metaphors or icons to represent abstract concepts
- Include "key takeaway" callout boxes
- Use accordion components for step-by-step processes

**IF COMPARATIVE** (vs, differences, options):
- Use side-by-side comparison tables or cards
- Highlight key differentiators with color coding
- Add "winner" or "best for" indicators if relevant

**IF CODE/TECHNICAL**:
- Syntax-highlighted code blocks with copy buttons
- Inline annotations explaining complex lines
- Collapsible sections for long code
- "Try it" or "Explanation" tabs

**IF LIST-BASED** (top X, rankings, steps):
- Numbered cards with hierarchy (larger for #1, smaller for later items)
- Progress indicators for sequential steps
- Visual icons for each item

**IF NARRATIVE/STORY**:
- Timeline or chapter-based navigation
- Quote callouts for key moments
- Image placeholders for visual breaks

**IF DATA-HEAVY** (statistics, research):
- Chart/graph components (even if placeholder)
- Stat callout boxes with large numbers
- Data table components with sortable columns

=== SPECIFIC REQUIREMENTS ===
1. **Interactivity**: Add at least 2 interactive elements (toggles, tabs, expandables, hovers)
2. **Hierarchy**: Use at least 3 levels of visual hierarchy (primary, secondary, tertiary)
3. **White Space**: Ensure content "breathes" - avoid cramped layouts
4. **Accessibility**: All interactive elements must have clear labels
5. **Mobile-First**: Design must work on small screens (single column when needed)

=== MARKDOWN CONTENT TO TRANSFORM ===
{markdown_content}

=== OUTPUT CONSTRAINTS ===
- Return ONLY the C1 DSL markup (no explanation, no preamble)
- The UI must render the COMPLETE content (never truncate or summarize)
- If the content is very long (>1500 words), use section-based navigation
- Every citation marker [1], [2] in the content MUST be preserved as clickable elements

=== CREATIVITY MANDATE ===
Don't just format the markdown - REIMAGINE it as an interface. Ask yourself: "If this were a premium app, how would it present this information?" Be bold with layout, use cards, grids, timelines, or custom components to make the content shine.
"""
    

    try:
        raw_dsl_string, status_code = await call_thesys_chat_api(thesys_meta_prompt)

        phase_end("thesys")  

        if 200 <= status_code < 300:
            print("[THESYS] Received C1 response.")
            return raw_dsl_string
        else:
            logger.warning("Thesys failed to generate C1 response. status=%s", status_code)
            return ""

    except Exception as e:
        logger.warning("Thesys UI generation error: %s", e)
        return ""


async def call_thesys_chat_api(prompt: str):
    """
    Uses the official OpenAI client pointed at the Thesys endpoint.
    This is the recommended approach per Thesys docs.
    """
    print("[THESYS_API] Calling Thesys via OpenAI-compatible client...")
    api_key = os.getenv("THESYS_API_KEY")
    if not api_key:
        print("[THESYS_API] THESYS_API_KEY is not configured.")
        return json.dumps({"error": "Server API key not configured."}), 500

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.thesys.dev/v1/embed"
        )

        completion = await with_timeout(
            client.chat.completions.create(
                model="c1/anthropic/claude-sonnet-4.6/v-20260331",
                messages=[{"role": "user", "content": prompt}],
            ),
            settings.THESYS_TIMEOUT_SECONDS,
            "Thesys UI generation",
            "Interactive UI generation is taking too long. The Markdown answer is still available.",
        )

        message_content = completion.choices[0].message.content

        if message_content is None:
            print("[THESYS_API] Empty response content.")
            return json.dumps({"error": "Empty response from Thesys."}), 500

        print(f"[THESYS_API] Response length: {len(message_content)} chars.")
        return str(message_content), 200

    except Exception as e:
        print(f"[THESYS_API] Unexpected error: {e}")
        return json.dumps({"error": "Unexpected server error.", "details": str(e)}), 500
    
    
async def _synthesize_answer_from_context(
    prompt: str, scraped_data: List[Dict[str, str]], context_package: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """Streams a cited answer from the retrieved source snippets."""
    
    phase_start("synthesis")  

    formatted_context = ""
    for i, item in enumerate(scraped_data, 1):
        formatted_context += f"[Source {i}: {item['source']}]\n{item['content']}\n\n"

    conversation_history = _format_context_for_prompt(context_package)

    system_prompt = f"""
You are an elite research analyst with a gift for synthesizing complex information into clear, actionable insights. Your answers are trusted by decision-makers because they're accurate, well-sourced, and easy to understand.

=== CONVERSATION CONTEXT ===
{conversation_history}

=== YOUR CORE MISSION ===
Answer the user's question by synthesizing information from the provided sources. Your answer should be THE definitive resource on this topic - comprehensive yet concise, authoritative yet accessible.

=== CRITICAL RULES ===

**RULE 1: SOURCE FIDELITY**
- Base your answer EXCLUSIVELY on the provided sources
- If sources conflict, acknowledge it: "Sources differ on this point: [1] suggests X, while [2] indicates Y"
- If sources are insufficient, be honest: "The provided sources don't contain information about [specific aspect]"
- NEVER invent information, even if you "know" it from your training

**RULE 2: SMART CITATION STRATEGY**
- Cite CLAIMS and FACTS, not every sentence
- Group related information under one citation: "Recent studies show three key findings: A, B, and C [1][2]"
- Don't cite common knowledge or transitional statements
- For significant claims, use multiple sources if available: [1][2][3]

**RULE 3: ANSWER STRUCTURE**
Follow this hierarchy based on question complexity:

**FOR SIMPLE FACTUAL QUESTIONS** (who, what, when, where):
- Direct answer in first sentence with citation
- 1-2 sentences of context
- Total: 2-4 sentences

**FOR EXPLANATORY QUESTIONS** (how, why):
- Brief overview (1 sentence)
- Main explanation (2-4 paragraphs)
- Key takeaway or implication
- Total: 200-400 words

**FOR COMPREHENSIVE QUESTIONS** (compare, analyze, list):
- Executive summary (2-3 sentences)
- Structured sections with headers
- Bullet points for key details
- Conclusion or recommendation
- Total: 400-600 words

**RULE 4: CONVERSATIONAL INTELLIGENCE**
- Reference previous context naturally: "As we discussed earlier regarding [topic]..."
- Use follow-up language: "This builds on the previous point about..."
- Don't repeat information already covered unless clarifying
- Adjust depth based on conversation progression (deeper for follow-ups)

**RULE 5: MARKDOWN MASTERY**
- Use **bold** for key terms (first mention only)
- Use headers (##) to break up long answers
- Use bullet points for lists of 3+ items
- Use > blockquotes for important definitions or quotes
- Use `code` formatting for technical terms, commands, or formulas

**RULE 6: QUALITY INDICATORS**
Your answer must have:
 A clear "answer" to the question in the first 2 sentences
 Logical flow (each paragraph connects to the next)
 Specific details, not vague generalities
 Citations that feel natural, not intrusive
 A sense of completeness (reader feels satisfied)


=== SNIPPET PROCESSING ===
You are receiving search result snippets. They are highly relevant but concise.
Use them to build a comprehensive answer. If a snippet is truncated, cite what is available.

=== HANDLING EDGE CASES ===

**IF sources are tangential but useful:**
"While the sources don't directly address [X], they provide related information about [Y] that may be helpful [1]"

**IF sources are outdated:**
"Based on available sources (dating from [timeframe]), the answer is [X] [1]. Note that this information may have changed"

**IF question has multiple interpretations:**
"Your question could mean [interpretation A] or [interpretation B]. I'll address both:..."

**IF answer requires nuance:**
Use phrases like "Generally...", "In most cases...", "However, there are exceptions..."

=== ANTI-PATTERNS (NEVER DO THIS) ===
 Starting with "Based on the sources provided..." (assumed)
 Ending with "I hope this helps!" (too casual)
 Apologizing ("Sorry, but...") - be confident or transparent
 Over-hedging ("might", "perhaps", "possibly" in every sentence)
 Bullet lists without context (always have a lead-in sentence)
 Walls of text (break into paragraphs of 3-5 sentences max)

=== YOUR TONE ===
Professional but approachable. You're a knowledgeable colleague, not a formal report. Use "you" when addressing the user. Vary sentence length for readability.
"""
    
    full_prompt = [
        system_prompt,
        "--- CONTEXT: SOURCES ---",
        formatted_context,
        "--- USER QUESTION ---",
        f"user: {prompt}",
    ]

    
    model = genai.GenerativeModel(model_name=settings.GEMINI_MODEL)

   
    try:
        response_stream = await with_timeout(
            model.generate_content_async(full_prompt, stream=True),
            settings.PROVIDER_TIMEOUT_SECONDS,
            "Gemini synthesis",
            "Answer synthesis is taking too long. Showing the retrieved source summary instead.",
        )
        async for chunk in iter_with_timeout(
            response_stream,
            settings.GEMINI_STREAM_TIMEOUT_SECONDS,
            "Gemini synthesis",
            "Answer synthesis stopped responding. Showing the retrieved source summary instead.",
        ):
            if chunk.text:
                yield chunk.text
    finally:
        phase_end("synthesis")




async def generate_and_stream_answer(
    prompt: str,
    path: str,
    session_id: str,
    turn_number: int,
    context_package: Dict[str, Any],
    user_id: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Streams the answer path, then stores the completed turn for memory."""
    total_start = time.perf_counter()

    sources_for_log = []
    images_for_log = []
    steps_for_log = []

    def step_event(message: str) -> Dict[str, Any]:
        steps_for_log.append(message)
        return {"event": "steps", "data": {"message": message}}

    try:
        full_markdown_response = ""
        provider_warnings_for_log = []

        def warning_event(message: str) -> Dict[str, Any]:
            provider_warnings_for_log.append(message)
            return {"event": "provider_warning", "data": {"message": message}}

        yield {"event": "analysis_complete", "data": {"path": path}}
        
        if path == "direct_answer":
            print("[ORCHESTRATOR] Executing direct answer path.")
            yield step_event("Generating answer...")
            
            conversation_history = _format_context_for_prompt(context_package)
            direct_prompt = f"Conversation History:\n{conversation_history}\n\nUser's Question: {prompt}"
            
            yield {"event": "synthesis_start", "data": {}}
            
            direct_model = genai.GenerativeModel(model_name=settings.GEMINI_MODEL)
            response_stream = await with_timeout(
                direct_model.generate_content_async(direct_prompt, stream=True),
                settings.PROVIDER_TIMEOUT_SECONDS,
                "Gemini direct answer",
                "The answer model is taking too long. Please try again.",
            )
            async for chunk in iter_with_timeout(
                response_stream,
                settings.GEMINI_STREAM_TIMEOUT_SECONDS,
                "Gemini direct answer",
                "The answer stream stopped responding. Please try again.",
            ):
                if chunk.text:
                    full_markdown_response += chunk.text
                    yield {"event": "markdown_chunk", "data": {"chunk": chunk.text}}

        else:
            print("[ORCHESTRATOR] Executing search path.")
            yield step_event("Searching the web...")

            image_task = asyncio.create_task(_get_images_from_tavily_async(prompt))

            tavily_response = await _search_tavily_async(prompt)
            tavily_results = tavily_response.get("results", [])
            tavily_instant_answer = tavily_response.get("answer")
            provider_error = tavily_response.get("provider_error")
            use_search_results = not provider_error

            if provider_error:
                yield warning_event(provider_error)
                path = "direct_answer"
                conversation_history = _format_context_for_prompt(context_package)
                fallback_prompt = f"Conversation History:\n{conversation_history}\n\nUser's Question: {prompt}"
                direct_model = genai.GenerativeModel(model_name=settings.GEMINI_MODEL)
                response_stream = await with_timeout(
                    direct_model.generate_content_async(fallback_prompt, stream=True),
                    settings.PROVIDER_TIMEOUT_SECONDS,
                    "Gemini fallback answer",
                    "The fallback answer model is taking too long. Please try again.",
                )
                async for chunk in iter_with_timeout(
                    response_stream,
                    settings.GEMINI_STREAM_TIMEOUT_SECONDS,
                    "Gemini fallback answer",
                    "The fallback answer stream stopped responding. Please try again.",
                ):
                    if chunk.text:
                        full_markdown_response += chunk.text
                        yield {"event": "markdown_chunk", "data": {"chunk": chunk.text}}
                images = await image_task
                if images:
                    images_for_log = images
                    yield {"event": "images", "data": {"images": images}}

            if use_search_results:
                if tavily_instant_answer:
                    yield step_event("Found quick results...")
                    instant_header = f"> **Quick Summary:** {tavily_instant_answer}\n\n---\n\n"
                    full_markdown_response += instant_header
                    yield {"event": "markdown_chunk", "data": {"chunk": instant_header}}

                images = await image_task
                if images:
                    logger.info("Found %s image results.", len(images))
                    images_for_log = images
                    yield {"event": "images", "data": {"images": images}}

                scraped_data = [
                    {
                        "source": res.get("url", ""),
                        "title": res.get("title", "Untitled"),
                        "content": res.get("content", "")
                    }
                    for res in tavily_results
                ]

                sources_for_log = [
                    {
                        "title": item["title"],
                        "url": item["source"],
                        "content": item["content"]
                    }
                    for item in scraped_data
                ]

                sources_for_ui = [
                    {"title": item["title"], "url": item["source"]}
                    for item in scraped_data
                ]
                yield {"event": "sources", "data": {"sources": sources_for_ui}}

                yield step_event("Synthesizing full cited answer...")
                yield {"event": "synthesis_start", "data": {}}

                try:
                    async for chunk in _synthesize_answer_from_context(prompt, scraped_data, context_package):
                        if chunk:
                            full_markdown_response += chunk
                            yield {"event": "markdown_chunk", "data": {"chunk": chunk}}
                except ProviderError as synthesis_error:
                    logger.warning(
                        "Using source-backed fallback after %s failed: %s",
                        synthesis_error.provider,
                        synthesis_error.public_message,
                    )
                    yield warning_event(synthesis_error.public_message)
                    fallback_answer = "\n\n---\n\n" + _build_snippet_fallback_answer(
                        prompt,
                        scraped_data,
                        None,
                    )
                    full_markdown_response += fallback_answer
                    yield {"event": "markdown_chunk", "data": {"chunk": fallback_answer}}

        if full_markdown_response.strip():
            yield step_event("Generating interactive UI...")
            print("[ORCHESTRATOR] Running UI generation and metadata.")

            async def _run_thesys():
                try:
                    return await generate_ui_spec_from_markdown(full_markdown_response, context_package)
                except Exception as exc:
                    logger.warning("Interactive UI fallback used: %s", exc)
                    return ""

            async def _run_metadata():
                try:
                    if turn_number == 1:
                        t, s, e = await with_timeout(
                            asyncio.gather(
                                _generate_chat_title(prompt),
                                _generate_summary(full_markdown_response),
                                _extract_entities(full_markdown_response),
                            ),
                            settings.METADATA_TIMEOUT_SECONDS,
                            "Gemini metadata",
                            "Response metadata generation is taking too long.",
                        )
                    else:
                        t = None
                        s, e = await with_timeout(
                            asyncio.gather(
                                _generate_summary(full_markdown_response),
                                _extract_entities(full_markdown_response),
                            ),
                            settings.METADATA_TIMEOUT_SECONDS,
                            "Gemini metadata",
                            "Response metadata generation is taking too long.",
                        )
                    return t, s, e
                except Exception as exc:
                    logger.warning("Metadata fallback used: %s", exc)
                    return (prompt[:50] if turn_number == 1 else None), "A response was generated.", []

            (raw_dsl_string, (title, summary, entities)) = await asyncio.gather(
                _run_thesys(),
                _run_metadata(),
            )

            print("[ORCHESTRATOR] UI generation and metadata complete.")

            if raw_dsl_string:
                yield {"event": "aui_dsl", "data": raw_dsl_string}
            else:
                yield warning_event("Interactive UI generation is temporarily unavailable. The Markdown answer is still available.")
            await asyncio.sleep(0.1)
            yield {"event": "turn_metadata", "data": {"summary": summary, "entities": entities}}

            log_data = {
                "user_id": user_id,
                "session_id": session_id,
                "turn_number": turn_number,
                "user_query": prompt,
                "response_summary": summary,
                "entities_mentioned": entities,
                "full_response_spec": raw_dsl_string,
                "sources_used": sources_for_log,
                "images": images_for_log,
                "steps": steps_for_log,
                "provider_warnings": provider_warnings_for_log,
                "execution_path": path,
                "created_at": datetime.now(timezone.utc),
                "full_markdown_response": full_markdown_response
            }
            if title:
                log_data["chat_title"] = title

            await _log_turn_to_db(log_data)

        else:
            yield {"event": "error", "data": {"message": "Failed to generate a valid response."}}

            
    except ProviderError as e:
        logger.warning("Provider error from %s: %s", e.provider, e.public_message)
        yield {"event": "error", "data": {"message": e.public_message, "provider": e.provider}}
    except Exception as e:
        logger.exception("Unexpected stream error: %s", e)
        yield {"event": "error", "data": {"message": "ARGON hit an unexpected server issue. Please try again."}}
    
    finally:
        total = time.perf_counter() - total_start     
        logger.info("Stream finished in %.2fs.", total)
        yield {"event": "finished", "data": {"message": "Stream completed."}}



async def stream_sse_formatter(
    event_generator: AsyncGenerator[Dict[str, Any], None]
) -> AsyncGenerator[str, None]:
    async for event in event_generator:
        event_name = event["event"]
        payload = event["data"]
        
        sse_message = f"event: {event_name}\n"

        if event_name == "aui_dsl":
            lines = payload.split('\n')
            for line in lines:
                sse_message += f"data: {line}\n"
        else:
            data_string = json.dumps(payload)
            sse_message += f"data: {data_string}\n"
            
        sse_message += "\n"
        yield sse_message
        await asyncio.sleep(0)
        

async def _generate_summary(markdown_content: str) -> str:
    """Uses a fast LLM to generate a one-sentence summary of the response."""
    print("  > [METADATA] Generating response summary...")
    try:
        model = genai.GenerativeModel(model_name=settings.GEMINI_METADATA_MODEL)
        prompt = f"""
        Analyze the following text, which is an AI-generated answer to a user's query.
        Your task is to create a very concise, one-sentence summary of the answer's main point.
        This summary will be used as conversational memory. Do not include any preamble.
        
        TEXT TO SUMMARIZE:
        ---
        {markdown_content}
        ---
        """
        response = await model.generate_content_async(prompt)
        summary = response.text.strip().replace('\n', ' ')
        print(f"    - Summary created: \"{summary}\"")
        return summary
    except Exception as e:
        print(f"[METADATA] Error generating summary: {e}")
        return "A response was generated." 

async def _extract_entities(markdown_content: str) -> List[str]:
    """Uses a fast LLM to extract key entities from the response."""
    print("  > [METADATA] Extracting key entities...")
    try:
        model = genai.GenerativeModel(
            model_name=settings.GEMINI_METADATA_MODEL,
            generation_config={"response_mime_type": "application/json"}
        )
        prompt = f"""
        Analyze the following text. Identify and extract the 3-5 most important proper nouns,
        concepts, or key terms. These entities will be used for contextual memory.
        
        You MUST respond with ONLY a valid JSON object with a single key "entities",
        which contains a list of the extracted entity strings.
        
        TEXT TO ANALYZE:
        ---
        {markdown_content}
        ---
        """
        response = await model.generate_content_async(prompt)
        result = json.loads(response.text)
        entities = result.get("entities", [])
        print(f"    - Entities extracted: {entities}")
        return entities
    except Exception as e:
        print(f"[METADATA] Error extracting entities: {e}")
        return [] 

async def _generate_chat_title(user_query: str) -> str:
    """
    Uses a fast LLM to generate a concise, high-quality, user-facing title
    for a new conversation, based on the user's first query.
    """
    print("  > [METADATA] Generating DEDICATED chat title for new session...")
    try:
        model = genai.GenerativeModel(model_name=settings.GEMINI_METADATA_MODEL)
        
        prompt = f"""
        Analyze the user's initial query. Your task is to create a concise, 3-to-6-word, user-facing title for the conversation that is about to begin. The title should accurately represent the user's primary intent.

        - User Query: "explain the theory of relativity and its impact on GPS" -> Title: Theory of Relativity & GPS
        - User Query: "write a short horror story about a foggy night on a lonely road" -> Title: Foggy Night Horror Story
        - User Query: "give me the merge sort code in python" -> Title: Merge Sort in Python

        Do NOT use quotation marks. Respond with ONLY the title.

        USER'S QUERY TO ANALYZE:
        ---
        "{user_query}"
        ---
        """
        response = await model.generate_content_async(prompt)
        title = response.text.strip().replace('\n', ' ')
        
        print(f"    - Dedicated title created: \"{title}\"")
        return title
        
    except Exception as e:
        print(f"[METADATA] Error generating chat title: {e}")

        return user_query[:50]
    

async def _log_turn_to_db(log_data: dict):
    """
    Updates the final turn data in MongoDB.
    Uses upsert=True for resilience: if the initial record somehow failed to be
    created, this will create it. Otherwise, it updates the existing placeholder.
    """
    if conversations_collection is None:
        print("[DB_LOG] Cannot log turn: conversations_collection is not available.")
        return
    
    try:

        query_filter = {
            "user_id": log_data["user_id"],
            "session_id": log_data["session_id"],
            "turn_number": log_data["turn_number"]
        }


        update_data = {"$set": log_data}

        await conversations_collection.update_one(query_filter, update_data, upsert=True)
        
        print(f"[DB_LOG] Successfully logged/updated turn {log_data['turn_number']} for session {log_data['session_id']}")
    except Exception as e:
        print(f"[DB_LOG] Failed to log/update turn to MongoDB: {e}")
