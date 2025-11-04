import os 
import json
import google.generativeai as genai
from typing import List

#stage 3
import asyncio
from tavily import AsyncTavilyClient
from typing import Dict, Any

#stage 4
from crawl4ai import AsyncWebCrawler

#stage 5
from typing import Dict, Any, AsyncGenerator, List 
from asgiref.sync import sync_to_async

# New Intelligent Classification Pipeline
import re
from datetime import datetime

# NLP function 
import spacy
from typing import Dict, Any

# Interactive UI
import httpx

#thesys
import html

#database
from datetime import datetime, timezone
from .db_config import conversations_collection



# Read the API key directly from the environment variable.
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    # Configure the library ONCE when the module is loaded.
    genai.configure(api_key=api_key)
    print("✅ [SERVICES] Google Generative AI configured successfully.")
except Exception as e:
    print(f"🚨 [SERVICES] Failed to configure Google Generative AI: {e}")



class GeminiError(Exception):
    """Custom exception for errors related to the Gemini API."""
    pass

# Contextual metadata extraction 

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


def extract_contextual_metadata(prompt: str) -> Dict[str, bool]:
    """
    Performs a rapid, computationally cheap analysis of the query's form
    to extract key contextual metadata before deeper processing.
    """
    print("✅ [CLASSIFIER STAGE 1] Extracting contextual metadata...")
    
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
    print("✅ [CLASSIFIER STAGE 2] spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("🚨 [CLASSIFIER STAGE 2] spaCy model not found. Please run 'python -m spacy download en_core_web_sm'")
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
    """(UPDATED FOR CONTEXT) Helper to get semantic classifications from the LLM."""
    
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
            model_name="gemini-2.5-flash", 
            generation_config={"response_mime_type": "application/json"}
        )
        response = model.generate_content(system_prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"🚨 [CLASSIFIER STAGE 2] Error during LLM classification: {e}")
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
    print("✅ [CLASSIFIER STAGE 2] Generating NLP features and scores...")
    
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
    print("✅ [CLASSIFIER STAGE 3] Making final routing decision...")

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

    # Apply the threshold to make the final decision
    if decision_score > DECISION_THRESHOLD:
        path = "search_required"
    else:
        path = "direct_answer"
        
    print(f"  > Final Path Decided: '{path}'")
    return path

async def get_intelligent_path(prompt: str, context_package: Dict[str, Any]) -> str:
    """(CORRECTED) The main orchestrator for the classification pipeline."""
    print("🚀 STARTING INTELLIGENT CLASSIFICATION PIPELINE 🚀")
    context_metadata = extract_contextual_metadata(prompt)
    
    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(
        None, generate_nlp_features_and_scores, prompt, context_metadata, context_package
    )
    
    final_path = make_routing_decision(scores)
    print(f"🏁 INTELLIGENT PIPELINE FINISHED. Final Path: {final_path} 🏁")
    
    return final_path




# stage 2
async def generate_search_queries(prompt: str, context_package: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    (REFACTORED AS ASYNC GENERATOR - CORRECTED VERSION)
    Yields `query_generated` events for each query, then yields a final
    `queries_complete` event with the full list.
    """
    conversation_history = _format_context_for_prompt(context_package)

    try:
        generation_config = {
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
        }
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash", 
            generation_config=generation_config,
        )

        system_prompt = f"""
You are an expert search query generation assistant. Your goal is to decompose the user's question into 3 to 5 simple, effective search queries.

**CRITICAL INSTRUCTION:** Analyze the provided conversation history. DO NOT generate queries for topics that have already been clearly answered or discussed. Focus your queries ONLY on the new information required by the user's latest question.

---
**CONVERSATION HISTORY:**
{conversation_history}
---
**USER'S CURRENT QUESTION:**
{prompt}
---

You must respond with ONLY a valid JSON object with a single key "queries", containing a list of strings. Do not answer the question or add commentary.
"""
        
        print(f"✅ [SERVICES] Sending prompt to Gemini for query generation: '{prompt}'")
        response = model.generate_content(system_prompt)
        response_json = json.loads(response.text)
        
        if "queries" not in response_json or not isinstance(response_json["queries"], list):
            raise GeminiError("Invalid JSON: 'queries' key is missing or not a list.")
            
        generated_queries = response_json["queries"]
        
        # LOGIC: YIELD each query individually 
        for query in generated_queries:
            yield {"event": "query_generated", "data": {"query": query}}
        
        #  YIELD the complete list as a special event 
        yield {"event": "queries_complete", "data": {"queries": generated_queries}}
        
        print(f"✅ [SERVICES] Successfully generated and yielded queries: {generated_queries}")

    except Exception as e:
        print(f"🚨 [SERVICES] An error occurred while generating search queries: {e}")
        yield {"event": "error", "data": {"message": f"Failed to generate search queries: {e}"}}

# stage 3

async def _search_tavily_async(query: str) -> List[Dict[str, Any]]:
    """
    An asynchronous helper function to perform a single search using the real Tavily API.
    
    Args:
        query: The search query string.

    Returns:
        A list of search result dictionaries from Tavily, or an empty list if an error occurs.
    """
    try:
        print(f"  > Starting search for: '{query}'")

        # Get the Tavily API key directly from the environment variables.

        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            # If the key is not found, log an error and fail gracefully for this task.
            print("🚨 TAVILY_API_KEY not found in environment variables.")
            return []
        
        # Initialize the Tavily client with the key.
        tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

        response = await tavily_client.search(
            query=query,
            search_depth="basic", # 'basic' is faster and sufficient for our needs
            max_results=5
        )
        
        print(f"  < Finished search for: '{query}'")
        return response.get("results", [])
    except Exception as e:
        # If any other error occurs during a single Tavily search, we log it

        print(f"  🚨 Error searching for '{query}': {e}")
        return []


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
            print("🚨 TAVILY_API_KEY not found in environment variables.")
            return []
        
        tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
        
        # The core of this function make the API call with include_images=True
        response = await tavily_client.search(
            query=query,
            search_depth="basic",    
            include_images=True,     
            max_results=15          
        )
        

        images = response.get("images", [])
        
        print(f"  < Finished IMAGE search for: '{query}'. Found {len(images)} images.")
        return images
        
    except Exception as e:
        print(f"  🚨 Error during IMAGE search for '{query}': {e}")
        return []


async def get_urls_from_queries(queries: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    (CORRECTED FOR LIVE STREAMING)
    Fetches results concurrently and yields `source_found` events in real-time
    as results come in.
    """
    print(f"✅ [SERVICES] Starting concurrent search for {len(queries)} queries...")
    tasks = [_search_tavily_async(query) for query in queries]
    
    all_urls = []
    source_count = 0

    # Using asyncio.as_completed to process tasks as they finish
    for future in asyncio.as_completed(tasks):
        results_from_one_query = await future
        urls_from_one_query = [result["url"] for result in results_from_one_query if "url" in result]
        
        # As soon as we get URLs from one query update the total count and yield
        if urls_from_one_query:
            all_urls.extend(urls_from_one_query)
            source_count = len(all_urls) # A running total
            yield {"event": "source_found", "data": {"count": source_count}}
            
    print(f"✅ [SERVICES] Found {len(all_urls)} total URLs (before de-duplication).")
    
    unique_urls = list(dict.fromkeys(all_urls))
    print(f"✅ [SERVICES] Found {len(unique_urls)} unique URLs.")

    MAX_URLS_TO_SCRAPE = 7
    limited_urls = unique_urls[:MAX_URLS_TO_SCRAPE]
    print(f"✅ [SERVICES] Limiting to {len(limited_urls)} URLs for scraping.")

    # Yield the final list of URLs to be used by the orchestrator
    yield {"event": "urls_complete", "data": {"urls": limited_urls}}


# stage 4
async def _scrape_one_url(crawler: AsyncWebCrawler, url: str) -> Dict[str, str]:
    """
    An asynchronous helper function to perform a single scrape using the real Crawl4AI library.
    This function is designed to be resilient.

    Args:
        crawler: An active instance of AsyncWebCrawler.
        url: The URL to scrape.

    Returns:
        A dictionary containing the source URL and the scraped markdown content.
        Returns an empty content string if scraping fails.
    """
    print(f"  > Starting scrape for: {url}")
    try:

        # The library uses its own internal default timeouts.
        result = await crawler.arun(url=url)
        
        # Check if the crawler returned valid content
        if result and result.markdown:
            content_length = len(result.markdown)
            print(f"  < Finished scrape for: {url} ({content_length} chars)")
            return {"source": url, "content": result.markdown}
        else:
            print(f"  ? No content found for: {url}")
            return {"source": url, "content": ""}
            
    except Exception as e:
        # This will catch any errors from the crawl4ai library itself.
        print(f"  🚨 Error scraping '{url}': {e}")
        return {"source": url, "content": ""}


async def scrape_urls_in_parallel(urls: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Takes a list of URLs and scrapes them all in parallel using Crawl4AI.
    It then filters out failed or empty scrapes.

    Args:
        urls: A list of URL strings to scrape.

    Returns:
        A list of dictionaries, each containing the 'source' URL and its
        'content' in Markdown format.
    """
    print(f"✅ [SERVICES] Starting parallel scrape for {len(urls)} URLs...")

    scraped_data = []

    async with AsyncWebCrawler() as crawler:
        #  1. Prepare the Tasks 
        tasks = []
        for url in urls:
            # Extract domain for the event
            domain = url.split('/')[2].replace('www.', '')
           
            yield {"event": "scraping_start", "data": {"domain": domain}}
            tasks.append(_scrape_one_url(crawler, url))

        results = await asyncio.gather(*tasks)

        #  3. Process and Filter Results 
        for result in results:
            # Check if the content is not empty or just whitespace.
            if result and result.get("content", "").strip():
                scraped_data.append(result)
            else:
                print(f"  - Discarding empty result for: {result.get('source')}")

        yield {"event": "scraping_complete", "data": {"total_scraped": len(scraped_data)}}
        print(f"✅ [SERVICES] Finished scraping. Successfully extracted content from {len(scraped_data)} out of {len(urls)} URLs.")

       
        yield {"event": "scraping_data_complete", "data": {"scraped_data": scraped_data}}

# thesys implementation 
async def generate_ui_spec_from_markdown(markdown_content: str, context_package: Dict[str, Any]) -> str:
    """(UPDATED FOR CONTEXT) The primary bridge to Thesys."""
    print("✅ [THESYS] Starting conversion of Markdown to UI Spec...")
    
    conversation_history = _format_context_for_prompt(context_package)

    thesys_meta_prompt = f"""
You are a world-class UI/UX architect specializing in transforming text into intuitive, visually engaging interfaces. Your goal is to create a UI that maximizes comprehension, engagement, and usability.

=== CORE DESIGN PRINCIPLES ===
1. **Clarity First**: The UI must make the information EASIER to understand than plain text
2. **Progressive Disclosure**: Show essential info first, hide complexity behind interactions
3. **Visual Hierarchy**: Use size, color, and spacing to guide the eye
4. **Scannable**: Users should grasp the structure in 2-3 seconds

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

        if 200 <= status_code < 300:
            print("✅ [THESYS] Successfully received raw C1 DSL string.")
            return raw_dsl_string
        else:
            print(f"🚨 [THESYS] Failed to generate UI Spec. Status: {status_code}")
            # Return an error string that the orchestrator can detect
            return "Error: The UI generation service failed to respond correctly."

    except Exception as e:
        print(f"🚨 [THESYS] A critical error occurred during UI generation: {e}")
        return f"Error: An error occurred during UI generation: {str(e)}"


async def call_thesys_chat_api(prompt: str):
    """
    (ENHANCED DEBUGGING VERSION)
    Acts as a secure proxy to the Thesys Chat Completions API.
    Now includes extensive logging to debug the exact response.
    """
    print("  > [THESYS_API] Calling Thesys Chat Completions API (Debug Mode)...")
    api_key = os.getenv("THESYS_API_KEY")
    if not api_key:
        error_msg = "🚨 FATAL: THESYS_API_KEY not found."
        print(error_msg)
        return json.dumps({"error": "Server API key not configured."}), 500

    api_url = "https://api.thesys.dev/v1/embed/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "c1-latest", "messages": [{"role": "user", "content": prompt}]}


    print(f"    - Target URL: {api_url}")
    print(f"    - Authorization Header: Bearer ...{api_key[-4:]}") # Log last 4 chars for verification
    print(f"    - Payload Length Sent: {len(json.dumps(payload))} characters")


    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(api_url, headers=headers, json=payload)

            
            raw_response_text = response.text
            print("\n    --- RECEIVED RESPONSE FROM THESYS ---")
            print(f"    - Status Code: {response.status_code}")
            print(f"    - Response Headers: {response.headers}")
            print(f"    - Raw Response Body: {raw_response_text}")
            print("    -------------------------------------\n")
           

            response.raise_for_status() 

            
            resp_json = json.loads(raw_response_text)
            message_content = resp_json.get("choices", [{}])[0].get("message", {}).get("content")

            if message_content is None:
                print("🚨 [THESYS_API] ERROR: 'message.content' field not found in the parsed JSON.")
                return json.dumps({"error": "Invalid response structure from Thesys API."}), 500

            # If we succeed, return the content
            return str(message_content), 200

    except httpx.HTTPStatusError as e:
        
        error_body = e.response.text
        print(f"🚨 [THESYS_API] HTTP Error Caught: {e.response.status_code}")
        print(f"   - Error Body from Exception: {error_body}")
        return json.dumps({"error": "Thesys API returned an error.", "details": error_body}), e.response.status_code
    
    except json.JSONDecodeError as e:
        
        print(f"🚨 [THESYS_API] JSON DECODE ERROR CAUGHT. The raw response body logged above is not valid JSON. Error: {e}")
        return json.dumps({"error": "Thesys returned a non-JSON response.", "details": raw_response_text}), 500

    except Exception as e:
        print(f"🚨 [THESYS_API] Unexpected Error: {e}")
        return json.dumps({"error": "Unexpected server error.", "details": str(e)}), 500#stage 5

async def _synthesize_answer_from_context(
    prompt: str, scraped_data: List[Dict[str, str]], context_package: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """(UPDATED FOR CONTEXT) Final RAG synthesis step."""
    
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
✓ A clear "answer" to the question in the first 2 sentences
✓ Logical flow (each paragraph connects to the next)
✓ Specific details, not vague generalities
✓ Citations that feel natural, not intrusive
✓ A sense of completeness (reader feels satisfied)

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
❌ Starting with "Based on the sources provided..." (assumed)
❌ Ending with "I hope this helps!" (too casual)
❌ Apologizing ("Sorry, but...") - be confident or transparent
❌ Over-hedging ("might", "perhaps", "possibly" in every sentence)
❌ Bullet lists without context (always have a lead-in sentence)
❌ Walls of text (break into paragraphs of 3-5 sentences max)

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

    
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")

   
    response_stream = model.generate_content(full_prompt, stream=True)
    for chunk in response_stream:
        if chunk.text:
            yield chunk.text

async def generate_and_stream_answer(
    prompt: str, path: str, session_id: str, turn_number: int, context_package: Dict[str, Any]
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    The main orchestrator. Handles both Search/RAG and Direct Answer paths,
    and uses Thesys to generate a final UI spec.
    """
    sources_for_log = []

    try:
        full_markdown_response = ""
        yield {"event": "analysis_complete", "data": {"path": path}}
        
        if path == "direct_answer":
            print("✅ [ORCHESTRATOR] Executing Direct Answer path.")
            yield {"event": "synthesis_start", "data": {}}

            yield {"event": "synthesis_start", "data": {}}

            yield {"event": "steps", "data": {"message": "Generating answer..."}}
            conversation_history = _format_context_for_prompt(context_package)
            direct_prompt = f"Conversation History:\n{conversation_history}\n\nUser's Question: {prompt}"
            model = genai.GenerativeModel(model_name="gemini-2.5-flash")
            response_stream = model.generate_content(direct_prompt, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    full_markdown_response += chunk.text
                    
        else: # path == "search_required"
            print("✅ [ORCHESTRATOR] Executing Search/RAG path.")
            yield {"event": "steps", "data": {"message": "Generating search queries..."}}
            
            queries = []
            async for event in generate_search_queries(prompt, context_package):
                if event['event'] == 'query_generated':
                    yield event  # Stream to frontend
                elif event['event'] == 'queries_complete':
                    queries = event['data']['queries']  # Extract final list
                elif event['event'] == 'error':
                    yield event
                    return

            if not queries:
                yield {"event": "error", "data": {"message": "Failed to generate search queries."}}
                return
            # Update the step message to reflect the parallel search

            for query in queries:
                yield {"event": "steps", "data": {"message": f"Searching for: \"{query}\""}}

            image_search_query = queries[0] 
            print(f"✅ [ORCHESTRATOR] Starting parallel search. Image query: '{image_search_query}'")

            text_urls_task = get_urls_from_queries(queries)
            image_results_task = _get_images_from_tavily_async(image_search_query)

            urls = []
            async for event in get_urls_from_queries(queries):
                if event['event'] == 'source_found':
                    yield event 
                elif event['event'] == 'urls_complete':
                    urls = event['data']['urls']  

            images = await _get_images_from_tavily_async(queries[0])
            print(f"✅ [ORCHESTRATOR] Parallel search complete. Found {len(urls)} text URLs and {len(images)} images.")


            if images: # Only send the event if we actually found images.
                print("✅ [ORCHESTRATOR] PREPARING TO STREAM 'images' EVENT.")
                print(f"  > DATA BEING SENT: {json.dumps({'images': images}, indent=2)}")
               

                yield {"event": "images", "data": {"images": images}}
                print("✅ [ORCHESTRATOR] 'images' event successfully yielded to stream.")
            else:
               
                print("🟡 [ORCHESTRATOR] No images found by Tavily. Skipping 'images' event.")



            yield {"event": "steps", "data": {"message": f"Reviewing {len(urls)} sources..."}}
            scraped_data = []
            async for event in scrape_urls_in_parallel(urls):
                if event['event'] == 'scraping_start':
                    yield event 
                elif event['event'] == 'scraping_complete':
                    yield event  
                elif event['event'] == 'scraping_data_complete':
                    scraped_data = event['data']['scraped_data']

            MIN_SOURCES_REQUIRED = 2
            if not scraped_data or len(scraped_data) < MIN_SOURCES_REQUIRED:
                yield {"event": "error", "data": {"message": f"Could only retrieve content from {len(scraped_data)} sources."}}
                return
            sources_for_log = [{"title": url.split('/')[2].replace('www.', ''), "url": url} for url in urls]
            yield {"event": "sources", "data": {"sources": sources_for_log}}
            yield {"event": "steps", "data": {"message": "Synthesizing the final answer..."}}
            yield {"event": "synthesis_start", "data": {}}
            async for chunk in _synthesize_answer_from_context(prompt, scraped_data, context_package):
                full_markdown_response += chunk

        if full_markdown_response.strip():
            yield {"event": "steps", "data": {"message": "Generating interactive UI..."}}
            raw_dsl_string = await generate_ui_spec_from_markdown(full_markdown_response, context_package)
            yield {"event": "aui_dsl", "data": raw_dsl_string}


            print("✅ [ORCHESTRATOR] Starting final metadata generation.")
            log_data = {}

            if turn_number == 1:
                print("  > Turn 1 detected. Generating DEDICATED title and summary.")
                # For the first turn, we generate everything: a title, a summary, and entities.
                title, summary, entities = await asyncio.gather(
                    _generate_chat_title(prompt),
                    _generate_summary(full_markdown_response),
                    _extract_entities(full_markdown_response)
                )
                
                
                print(f"    - Generated Title (for DB): '{title}'")
                print(f"    - Generated Summary (for context): '{summary}'")
                print(f"    - Generated Entities (for context): {entities}")
               

                
                metadata_payload = {"summary": summary, "entities": entities}
                yield {"event": "turn_metadata", "data": metadata_payload}
                
               
                log_data = {
                    "session_id": session_id,
                    "turn_number": turn_number,
                    "user_query": prompt,
                    "chat_title": title, 
                    "response_summary": summary, 
                    "entities_mentioned": entities,
                    "full_response_spec": raw_dsl_string,
                    "sources_used": sources_for_log,
                    "execution_path": path,
                    "created_at": datetime.now(timezone.utc)
                }
            else: 
                print(f"  > Turn {turn_number} detected. Generating summary for context only.")
                
                summary, entities = await asyncio.gather(
                    _generate_summary(full_markdown_response),
                    _extract_entities(full_markdown_response)
                )

                
                print(f"    - Generated Summary (for context): '{summary}'")
                print(f"    - Generated Entities (for context): {entities}")
                

                metadata_payload = {"summary": summary, "entities": entities}
                yield {"event": "turn_metadata", "data": metadata_payload}
                
                log_data = {
                    "session_id": session_id,
                    "turn_number": turn_number,
                    "user_query": prompt,
                    # NO chat_title field for subsequent turns
                    "response_summary": summary,
                    "entities_mentioned": entities,
                    "full_response_spec": raw_dsl_string,
                    "sources_used": sources_for_log,
                    "execution_path": path,
                    "created_at": datetime.now(timezone.utc)
                }

            # DEBUG LOG BEFORE DB WRITE
            print("  > Data prepared for database logging:")
            import pprint
            pprint.pprint(log_data)
            #  END OF DEBUG LOG 

            await _log_turn_to_db(log_data)
            

        else:
            yield {"event": "error", "data": {"message": "Failed to generate a response."}}

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"event": "error", "data": {"message": f"An unexpected error occurred: {str(e)}"}}
    
    finally:
        print("✅ [ORCHESTRATOR] Stream finished.")
        yield {"event": "finished", "data": {"message": "Stream completed."}}


async def stream_sse_formatter(
    event_generator: AsyncGenerator[Dict[str, Any], None]
) -> AsyncGenerator[str, None]:
    """
    (CORRECT MULTI-LINE FORMATTER)
    Wraps an async generator and formats its dictionary yields into
    Server-Sent Event (SSE) strings. Correctly handles multi-line data.
    """
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



async def _generate_summary(markdown_content: str) -> str:
    """Uses a fast LLM to generate a one-sentence summary of the response."""
    print("  > [METADATA] Generating response summary...")
    try:
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
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
        print(f"    - 🚨 Error generating summary: {e}")
        return "A response was generated." 

async def _extract_entities(markdown_content: str) -> List[str]:
    """Uses a fast LLM to extract key entities from the response."""
    print("  > [METADATA] Extracting key entities...")
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
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
        print(f"    - 🚨 Error extracting entities: {e}")
        return [] 

async def _generate_chat_title(user_query: str) -> str:
    """
    Uses a fast LLM to generate a concise, high-quality, user-facing title
    for a new conversation, based on the user's first query.
    """
    print("  > [METADATA] Generating DEDICATED chat title for new session...")
    try:
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        
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
        print(f"    - 🚨 Error generating dedicated title: {e}")

        return user_query[:50]
    

async def _log_turn_to_db(log_data: dict):
    """
    Updates the final turn data in MongoDB.
    Uses upsert=True for resilience: if the initial record somehow failed to be
    created, this will create it. Otherwise, it updates the existing placeholder.
    """
    if conversations_collection is None:
        print("🚨 [DB_LOG] Cannot log turn: conversations_collection is not available.")
        return
    
    try:

        query_filter = {
            "session_id": log_data["session_id"],
            "turn_number": log_data["turn_number"]
        }


        update_data = {"$set": log_data}

        await conversations_collection.update_one(query_filter, update_data, upsert=True)
        
        print(f"✅ [DB_LOG] Successfully logged/updated turn {log_data['turn_number']} for session {log_data['session_id']}")
    except Exception as e:
        print(f"🚨 [DB_LOG] Failed to log/update turn to MongoDB: {e}")