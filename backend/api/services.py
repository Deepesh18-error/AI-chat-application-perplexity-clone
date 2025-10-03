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


# stage 5 functions

def decide_query_path(prompt: str) -> str:
    """
    Uses a fast LLM to decide if a prompt requires a web search or can be answered directly.

    This is a synchronous function designed for a quick, blocking decision.

    Args:
        prompt: The user's input prompt.

    Returns:
        A string: 'search_required' or 'direct_answer'. Defaults to 'search_required' on failure.
    """
    print(f"✅ [SERVICES] STAGE 5.1: Deciding query path for: '{prompt}'")
    try:
        # 1. Select the fast model and configure for JSON output
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )

        # 2. Craft the classification system prompt
        system_prompt = """
        You are an expert query classifier. Your task is to determine if a user's prompt requires a real-time web search to answer accurately or if it can be answered directly by a large language model.

        Categorize the prompt into one of two paths:
        1.  `search_required`: For questions about recent events, specific facts, products, people, companies, or any topic that requires up-to-date, external information.
        2.  `direct_answer`: For self-contained questions like math problems, logic puzzles, creative writing requests, coding tasks, summarization of provided text, or general knowledge questions where real-time data is not essential.

        You must respond with ONLY a valid JSON object with a single key "path" and one of the two string values.
        Example: {"path": "search_required"}
        """
        
        # 3. Make the API call
        response = model.generate_content([system_prompt, "user: ", prompt])
        
        # 4. Parse and validate the response
        decision = json.loads(response.text)
        path = decision.get("path")
        
        if path not in ["search_required", "direct_answer"]:
            # Handle cases where the LLM returns an invalid value
            print(f"🚨 [SERVICES] Invalid path value received: '{path}'. Defaulting to search.")
            return "search_required"
            
        print(f"✅ [SERVICES] Path decided: '{path}'")
        return path

    except Exception as e:
        # 5. Fail-safe: If anything goes wrong, default to the more robust search path
        print(f"🚨 [SERVICES] An error occurred in decide_query_path: {e}. Defaulting to 'search_required'.")
        return "search_required"
    

decide_query_path_async = sync_to_async(decide_query_path, thread_sensitive=True)
    


# stage 2
def generate_search_queries(prompt: str) -> List[str]:
    """
    Takes a user's natural language prompt and uses the Gemini API
    to generate a list of targeted search queries.
    """
    try:
        #  Define the Generation Configuration
        generation_config = {
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
        }

        #  Select the Model (Use the correct, stable model name) 
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash", 
            generation_config=generation_config,
        )

        #  Craft the Prompt
        system_prompt = """
        You are an expert search query generation assistant. Your sole purpose is to
        analyze the user's question and decompose it into 3 to 5 simple, effective
        search engine queries that will gather the necessary information to provide a
        comprehensive answer.

        Do not answer the question yourself. Do not add any commentary or explanation.

        You must respond with ONLY a valid JSON object. The JSON object must have a
        single key named "queries", which contains a list of the generated search
        query strings.
        """
        full_prompt = [system_prompt, "user: ", prompt]

        #  Execute the API Call 
        print(f"✅ [SERVICES] Sending prompt to Gemini: '{prompt}'")
        response = model.generate_content(full_prompt)

        #  Parse and Validate the Response 
        response_json = json.loads(response.text)
        
        if "queries" not in response_json or not isinstance(response_json["queries"], list):
            raise GeminiError("Invalid JSON response format from Gemini: 'queries' key is missing or not a list.")
            
        queries = response_json["queries"]
        
        if not all(isinstance(q, str) for q in queries):
            raise GeminiError("Invalid JSON response format from Gemini: not all items in 'queries' are strings.")

        print(f"✅ [SERVICES] Successfully received and parsed queries from Gemini: {queries}")
        return queries

    except Exception as e:
        print(f"🚨 [SERVICES] An error occurred while calling the Gemini API: {e}")
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

#stage 5

async def _synthesize_answer_from_context(
    prompt: str, scraped_data: List[Dict[str, str]]
) -> AsyncGenerator[str, None]:
    """
    (Helper for the Orchestrator)
    The final RAG synthesis step. Takes the scraped context and generates the
    final cited answer, streaming the tokens.
    """
    # 1. Format the context for the LLM
    formatted_context = ""
    for i, item in enumerate(scraped_data, 1):
        formatted_context += f"[Source {i}: {item['source']}]\n{item['content']}\n\n"

    # 2. Craft the "Mega-Prompt"
    system_prompt = """
    You are a world-class AI research assistant. Your purpose is to answer the user's question with a comprehensive, well-structured, and factual response.

    Follow these instructions precisely:
    1.  **Analyze the User's Question:** Understand the core intent of the user's prompt.
    2.  **Synthesize from Sources:** Base your answer **exclusively** on the information provided in the numbered sources. Do not use any outside knowledge.
    3.  **Cite Everything:** You MUST cite every piece of information. Add a citation marker, like [1], [2], etc., at the end of each sentence or claim that is supported by a source. The number must correspond to the source number provided. You can use multiple citations for a single sentence, like [1][2].
    4.  **Format Beautifully:** Structure your answer using Markdown for clarity. Use headers, bullet points, and bold text to create a readable and organized response.
    5.  **Handle Insufficient Information:** If the provided sources do not contain enough information to answer the question, you must explicitly state: "I could not find enough information in the provided sources to answer this question." Do not try to make up an answer.
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


async def generate_and_stream_answer(
    prompt: str, path: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    The main orchestrator. Handles both Search/RAG and Direct Answer paths,
    yielding structured event dictionaries for the frontend.
    """
    try:
        # --- PATH 1: Direct Answer ---
        if path == "direct_answer":
            print("✅ [ORCHESTRATOR] Executing Direct Answer path.")
            yield {"event": "steps", "data": {"message": "Generating answer..."}}

            model = genai.GenerativeModel(model_name="gemini-2.5-flash")
            response_stream = model.generate_content(prompt, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    yield {"event": "token", "data": {"token": chunk.text}}
        
        # --- PATH 2: Full RAG Pipeline ---
        else: # path == "search_required"
            print("✅ [ORCHESTRATOR] Executing Search/RAG path.")
            
            # Step 1: Generate Search Queries
            yield {"event": "steps", "data": {"message": "Generating search queries..."}}
            queries = generate_search_queries(prompt)
            yield {"event": "search_queries", "data": {"queries": queries}}
            
            # Step 2: Get URLs (now limited)
            yield {"event": "steps", "data": {"message": "Searching the web..."}}
            urls = await get_urls_from_queries(queries)
            
            # Step 3: Scrape Content
            yield {"event": "steps", "data": {"message": f"Reviewing {len(urls)} sources..."}}
            scraped_data = await scrape_urls_in_parallel(urls)

            # Step 4: Edge Case Handling
            MIN_SOURCES_REQUIRED = 3
            if not scraped_data or len(scraped_data) < MIN_SOURCES_REQUIRED:
                print(f"🚨 [ORCHESTRATOR] Scraped too few sources ({len(scraped_data)}). Aborting.")
                yield {
                    "event": "error",
                    "data": {"message": f"I could not retrieve enough information from the web to provide a reliable answer. Only found {len(scraped_data)} sources."}
                }
                # The 'return' statement ends the generator function here.
                return

            # Step 5: Yield Source Information for the UI
            sources = [{"title": url.split('/')[2].replace('www.', ''), "url": url} for url in urls]
            yield {"event": "sources", "data": {"sources": sources}}

            # Step 6: Synthesize and stream the final answer
            yield {"event": "steps", "data": {"message": "Synthesizing the final answer..."}}
            
            async for token in _synthesize_answer_from_context(prompt, scraped_data):
                yield {"event": "token", "data": {"token": token}}

    except Exception as e:
        print(f"🚨 [ORCHESTRATOR] A critical error occurred: {e}")
        yield {
            "event": "error",
            "data": {"message": f"An unexpected error occurred during processing: {e}"}
        }
    
    finally:
        # Signal the end of the stream for all paths
        print("✅ [ORCHESTRATOR] Stream finished.")
        yield {"event": "finished", "data": {"message": "Stream completed."}}


async def stream_sse_formatter(
    event_generator: AsyncGenerator[Dict[str, Any], None]
) -> AsyncGenerator[str, None]:
    """
    Wraps an async generator that yields dictionaries and formats them
    into Server-Sent Event (SSE) strings for the client.
    """

    async for event in event_generator:
        event_name = event["event"]
        data = json.dumps(event["data"])
        # Format as an SSE message: event: <name>\ndata: <json>\n\n
        yield f"event: {event_name}\ndata: {data}\n\n"