import asyncio
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import services
from .services import GeminiError


@api_view(["POST"])
def generate_view(request):
    """
    API endpoint that orchestrates the entire data gathering pipeline:
    Prompt -> Queries -> URLs -> Scraped Content.

    Pipeline:
    1. (Input) Receives and validates the user's prompt.
    2. (Stage 2) Generates search queries using an LLM.
    3. (Stage 3) Fetches relevant URLs concurrently.
    4. (Stage 4) Scrapes the content from those URLs in parallel.
    5. (Output) For testing, returns the final structured, scraped data.
    """
    try:
        prompt = request.data.get("prompt")
        if not prompt or not isinstance(prompt, str):
            return Response(
                {"error": "A valid 'prompt' string is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        #  STAGE 2: QUERY GENERATION 
        print(f"✅ [VIEW] STAGE 2: Generating search queries for: '{prompt}'")
        queries = services.generate_search_queries(prompt)
        print(f"✅ [VIEW] STAGE 2: Generated {len(queries)} queries: {queries}")

        #  STAGE 3: CONCURRENT URL RETRIEVAL 
        print(f"✅ [VIEW] STAGE 3: Fetching URLs for {len(queries)} queries...")
        urls = asyncio.run(services.get_urls_from_queries(queries))
        print(f"✅ [VIEW] STAGE 3: Retrieved {len(urls)} unique URLs.")
        
        #  STAGE 4: PARALLEL CONTENT SCRAPING 
        print(f"✅ [VIEW] STAGE 4: Scraping content for {len(urls)} URLs...")
        scraped_data = asyncio.run(services.scrape_urls_in_parallel(urls))
        print(f"✅ [VIEW] STAGE 4: Successfully scraped {len(scraped_data)} pages.")


        #  FINAL RESPONSE (FOR STAGE 4 TESTING) 
        # The API now returns the fully processed, structured data.
        response_data = {"scraped_data": scraped_data}

        return Response(response_data, status=status.HTTP_200_OK)

    except GeminiError as e:
        # Handle errors specifically from the query generation stage
        print(f"🚨 [VIEW] A GeminiError occurred: {e}")
        return Response(
            {"error": f"An error occurred with the AI service: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    except Exception as e:
        # Catch any other unexpected errors from any stage
        print(f"🚨 [VIEW] An unexpected error occurred in the pipeline: {e}")
        return Response(
            {"error": f"An internal server error occurred: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )