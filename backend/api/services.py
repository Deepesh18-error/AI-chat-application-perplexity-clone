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
# This is the most direct and reliable way.
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


# Load the spaCy model once when the module is loaded for efficiency
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ [CLASSIFIER STAGE 2] spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("🚨 [CLASSIFIER STAGE 2] spaCy model not found. Please run 'python -m spacy download en_core_web_sm'")
    nlp = None

# Define the programmatic mapping from LLM classifications to numerical scores
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
    
    # Simple verb extraction (find the root verb of the main clause)
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
    # CRITICAL: This weight is NEGATIVE. High context dependency strongly
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
def generate_search_queries(prompt: str, context_package: Dict[str, Any]) -> List[str]:
    """
    Takes a user's natural language prompt and uses the Gemini API
    to generate a list of targeted search queries.
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
            
        queries = response_json["queries"]
        print(f"✅ [SERVICES] Successfully received context-aware queries: {queries}")
        return queries

    except Exception as e:
        print(f"🚨 [SERVICES] An error occurred while generating search queries: {e}")
        raise GeminiError(f"Failed to generate search queries: {e}")


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
        # This is more direct and robust than using the Django settings object here.
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            # If the key is not found, log an error and fail gracefully for this task.
            print("🚨 TAVILY_API_KEY not found in environment variables.")
            return []
        
        # Initialize the Tavily client with the key.
        tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
        #  END OF FIX 
        
        # Use the async client's search method.
        # We limit results to a reasonable number to avoid overwhelming the next stage.
        response = await tavily_client.search(
            query=query,
            search_depth="basic", # 'basic' is faster and sufficient for our needs
            max_results=5
        )
        
        print(f"  < Finished search for: '{query}'")
        return response.get("results", [])
    except Exception as e:
        # If any other error occurs during a single Tavily search, we log it
        # but return an empty list. This makes our parallel processing more resilient.
        print(f"  🚨 Error searching for '{query}': {e}")
        return []


async def get_urls_from_queries(queries: List[str]) -> List[str]:
    """
    Takes a list of search queries, fetches results concurrently, and returns
    a de-duplicated and LIMITED list of URLs.
    """
    print(f"✅ [SERVICES] Starting concurrent search for {len(queries)} queries...")
    tasks = [_search_tavily_async(query) for query in queries]
    all_results = await asyncio.gather(*tasks)
    flat_results = [item for sublist in all_results for item in sublist]
    urls = [result["url"] for result in flat_results if "url" in result]
    print(f"✅ [SERVICES] Found {len(urls)} total URLs (before de-duplication).")
    
    unique_urls = list(dict.fromkeys(urls))
    print(f"✅ [SERVICES] Found {len(unique_urls)} unique URLs.")

    # --- THIS IS THE NEW OPTIMIZATION ---
    MAX_URLS_TO_SCRAPE = 7
    limited_urls = unique_urls[:MAX_URLS_TO_SCRAPE]
    print(f"✅ [SERVICES] Limiting to {len(limited_urls)} URLs for scraping.")
    # ------------------------------------

    return limited_urls

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
        # The arun method is called with only the URL.
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


async def scrape_urls_in_parallel(urls: List[str]) -> List[Dict[str, str]]:
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
    # Use a single crawler instance for all jobs for efficiency.
    # The 'async with' block ensures the browser resources are cleaned up properly.
    async with AsyncWebCrawler() as crawler:
        #  1. Prepare the Tasks 
        tasks = [_scrape_one_url(crawler, url) for url in urls]

        #  2. Execute Concurrently 
        # asyncio.gather runs all scraping jobs at the same time.
        results = await asyncio.gather(*tasks)

        #  3. Process and Filter Results 
        for result in results:
            # Check if the content is not empty or just whitespace.
            if result and result.get("content", "").strip():
                scraped_data.append(result)
            else:
                print(f"  - Discarding empty result for: {result.get('source')}")

    print(f"✅ [SERVICES] Finished scraping. Successfully extracted content from {len(scraped_data)} out of {len(urls)} URLs.")
    return scraped_data

# thesys implementation 
async def generate_ui_spec_from_markdown(markdown_content: str, context_package: Dict[str, Any]) -> str:
    """(UPDATED FOR CONTEXT) The primary bridge to Thesys."""
    print("✅ [THESYS] Starting conversion of Markdown to UI Spec...")
    
    conversation_history = _format_context_for_prompt(context_package)

    thesys_meta_prompt = f"""
You are an expert UI/UX designer. You will be given a Markdown document and the preceding conversation history. Your sole task is to transform the Markdown content into the best possible rich, interactive UI.

Use the conversation history to inform your design. For example, if this is a follow-up question, the UI can be structured as a continuation of the previous topic.

---
**CONVERSATION HISTORY (for context):**
{conversation_history}
---
**MARKDOWN CONTENT TO TRANSFORM:**
{markdown_content}
---
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

# In services.py, replace the existing call_thesys_chat_api function

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

    # --- NEW LOGGING: Log the request we are sending ---
    print(f"    - Target URL: {api_url}")
    print(f"    - Authorization Header: Bearer ...{api_key[-4:]}") # Log last 4 chars for verification
    print(f"    - Payload Length Sent: {len(json.dumps(payload))} characters")
    # Uncomment the next line ONLY for intense debugging, as it can be very long
    # print(f"    - Full Payload Sent: {json.dumps(payload, indent=2)}")
    # --- END OF NEW LOGGING ---

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(api_url, headers=headers, json=payload)

            # --- CRITICAL LOGGING: Log the raw response BEFORE any parsing ---
            raw_response_text = response.text
            print("\n    --- RECEIVED RESPONSE FROM THESYS ---")
            print(f"    - Status Code: {response.status_code}")
            print(f"    - Response Headers: {response.headers}")
            print(f"    - Raw Response Body: {raw_response_text}")
            print("    -------------------------------------\n")
            # --- END OF CRITICAL LOGGING ---

            response.raise_for_status() # Still useful to catch 4xx/5xx errors

            # Now, attempt to parse the raw text we just logged
            resp_json = json.loads(raw_response_text)
            message_content = resp_json.get("choices", [{}])[0].get("message", {}).get("content")

            if message_content is None:
                print("🚨 [THESYS_API] ERROR: 'message.content' field not found in the parsed JSON.")
                return json.dumps({"error": "Invalid response structure from Thesys API."}), 500

            # If we succeed, return the content
            return str(message_content), 200

    except httpx.HTTPStatusError as e:
        # This will now have more context because we logged the body above
        error_body = e.response.text
        print(f"🚨 [THESYS_API] HTTP Error Caught: {e.response.status_code}")
        print(f"   - Error Body from Exception: {error_body}")
        return json.dumps({"error": "Thesys API returned an error.", "details": error_body}), e.response.status_code
    
    except json.JSONDecodeError as e:
        # This is the error we were seeing. Now we know exactly what text caused it.
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
You are a world-class AI research assistant. Your purpose is to answer the user's question with a comprehensive, well-structured, and factual response based ONLY on the provided sources.

**Conversation History (for context):**
{conversation_history}

**Instructions:**
1.  **Synthesize from Sources:** Base your answer **exclusively** on the information in the numbered sources below.
2.  **Cite Everything:** Add a citation marker, like [1], [2], at the end of each sentence or claim.
3.  **Be Conversational:** Use the conversation history to make your answer feel natural. You can refer to previously discussed topics (e.g., "Building on our earlier discussion...").
4.  **Format Beautifully:** Use Markdown (headers, lists, bold text).
5.  **Handle Insufficient Information:** If the sources don't contain the answer, state that clearly.
"""
    full_prompt = [
        system_prompt,
        "--- CONTEXT: SOURCES ---",
        formatted_context,
        "--- USER QUESTION ---",
        f"user: {prompt}",
    ]

    # 3. Select a powerful model for synthesis
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")

    # 4. Initiate the streaming call and yield tokens
    response_stream = model.generate_content(full_prompt, stream=True)
    for chunk in response_stream:
        if chunk.text:
            yield chunk.text


# services.py

# services.py

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
        
        if path == "direct_answer":
            print("✅ [ORCHESTRATOR] Executing Direct Answer path.")
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
            # --- FIX: Pass context_package to the query generator ---
            queries = generate_search_queries(prompt, context_package)
            # --- END OF FIX ---
            yield {"event": "steps", "data": {"message": "Searching the web..."}}
            urls = await get_urls_from_queries(queries)
            yield {"event": "steps", "data": {"message": f"Reviewing {len(urls)} sources..."}}
            scraped_data = await scrape_urls_in_parallel(urls)
            MIN_SOURCES_REQUIRED = 2
            if not scraped_data or len(scraped_data) < MIN_SOURCES_REQUIRED:
                yield {"event": "error", "data": {"message": f"Could only retrieve content from {len(scraped_data)} sources."}}
                return
            sources_for_log = [{"title": url.split('/')[2].replace('www.', ''), "url": url} for url in urls]
            yield {"event": "sources", "data": {"sources": sources_for_log}}
            yield {"event": "steps", "data": {"message": "Synthesizing the final answer..."}}
            async for chunk in _synthesize_answer_from_context(prompt, scraped_data, context_package):
                full_markdown_response += chunk

        if full_markdown_response.strip():
            yield {"event": "steps", "data": {"message": "Generating interactive UI..."}}
            raw_dsl_string = await generate_ui_spec_from_markdown(full_markdown_response, context_package)
            yield {"event": "aui_dsl", "data": raw_dsl_string}

            summary, entities = await asyncio.gather(
                _generate_summary(full_markdown_response),
                _extract_entities(full_markdown_response)
            )

            metadata_payload = {"summary": summary, "entities": entities}
            yield {"event": "turn_metadata", "data": metadata_payload}
            
            log_data = {
                "session_id": session_id,
                "turn_number": turn_number,
                "user_query": prompt,
                "response_summary": summary,
                "entities_mentioned": entities,
                "sources_used": sources_for_log,
                "execution_path": path,
                "created_at": datetime.now(timezone.utc)
            }
            asyncio.create_task(_log_turn_to_db(log_data))
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
            # For the multi-line DSL string, we split it by newline
            # and prepend "data: " to each line.
            lines = payload.split('\n')
            for line in lines:
                sse_message += f"data: {line}\n"
        else:
            # For all other single-object events, we serialize to JSON.
            data_string = json.dumps(payload)
            sse_message += f"data: {data_string}\n"
            
        # Terminate the message with an extra newline.
        sse_message += "\n"
        
        yield sse_message


# backend/api/services.py

# ... after the _format_context_for_prompt function

# --- NEW: Helper functions for Phase 3 ---

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
        return "A response was generated." # Return a generic fallback

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
        return [] # Return an empty list on failure
# --- END OF NEW HELPERS ---

async def _log_turn_to_db(log_data: Dict[str, Any]):
    """(CORRECTED) Logs a single conversational turn to MongoDB with a safe check."""
    # --- FIX: Compare with None instead of using a truthiness test ---
    if conversations_collection is None:
        print("🚨 [DB_LOG] Cannot log turn: conversations_collection is not available.")
        return
    # --- END OF FIX ---
    
    try:
        await conversations_collection.insert_one(log_data)
        print(f"✅ [DB_LOG] Successfully logged turn {log_data['turn_number']} for session {log_data['session_id']}")
    except Exception as e:
        print(f"🚨 [DB_LOG] Failed to log turn to MongoDB: {e}")