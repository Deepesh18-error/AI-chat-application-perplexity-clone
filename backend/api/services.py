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
    Takes a list of search queries and fetches the search results for all of
    them concurrently using the Tavily Search API.

    Args:
        queries: A list of search query strings.

    Returns:
        A de-duplicated list of URLs relevant to the search queries.
    """
    print(f"✅ [SERVICES] Starting concurrent search for {len(queries)} queries...")
    
    # 1. Prepare the Tasks
    # Create a list of "tasks" (coroutines) to run. Each task is a call
    # to our helper function with a different query.
    tasks = [_search_tavily_async(query) for query in queries]

    #  2. Execute Concurrently 
    # asyncio.gather() runs all the tasks in the list at the same time.
    # It waits until the last one is finished.
    # all_results will be a list of lists of dictionaries.
    all_results = await asyncio.gather(*tasks)

    #  3. Consolidate and Sanitize 
    # Flatten the list of lists into a single list of result dictionaries.
    flat_results = [item for sublist in all_results for item in sublist]

    # Extract just the URLs from the results.
    urls = [result["url"] for result in flat_results if "url" in result]

    print(f"✅ [SERVICES] Found {len(urls)} total URLs (before de-duplication).")
    
    # De-duplicate the list of URLs while preserving order
    # Using a set is faster for very large lists, but dict.fromkeys is a common
    # and readable way to preserve order.
    unique_urls = list(dict.fromkeys(urls))
    
    print(f"✅ [SERVICES] Returning {len(unique_urls)} unique URLs.")
    return unique_urls

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