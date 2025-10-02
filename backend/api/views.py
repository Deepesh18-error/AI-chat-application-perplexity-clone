import asyncio 
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import services
from .services import GeminiError


@api_view(["POST"])
def generate_view(request):
    """
    API endpoint that orchestrates the entire query -> retrieval pipeline.

    Pipeline:
    1. Receives and validates the user's prompt.
    2. (Stage 2) Delegates to the services layer to generate search queries.
    3. (Stage 3) Delegates again to the services layer to fetch URLs concurrently.
    4. For testing, returns the final list of unique URLs.
    """
    try:
        prompt = request.data.get("prompt")
        if not prompt or not isinstance(prompt, str):
            return Response(
                {"error": "A valid 'prompt' string is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # STAGE 2: QUERY GENERATION 
        print(f"✅ [VIEW] STAGE 2: Generating search queries for: '{prompt}'")
        queries = services.generate_search_queries(prompt)
        print(f"✅ [VIEW] STAGE 2: Generated {len(queries)} queries.")

        # STAGE 3: CONCURRENT URL RETRIEVAL 
        print(f"✅ [VIEW] STAGE 3: Fetching URLs for {len(queries)} queries...")
        # This is the "bridge" from synchronous to asynchronous code.
        # asyncio.run() executes our async function and waits for the result.
        urls = asyncio.run(services.get_urls_from_queries(queries))
        print(f"✅ [VIEW] STAGE 3: Retrieved {len(urls)} unique URLs.")


        # FINAL RESPONSE (FOR STAGE 3 TESTING) 
        # We now return the list of URLs instead of the queries.
        response_data = {"urls": urls}

        return Response(response_data, status=status.HTTP_200_OK)

    except GeminiError as e:
        # Handle errors specifically from the query generation stage
        print(f"🚨 [VIEW] A GeminiError occurred: {e}")
        return Response(
            {"error": f"An error occurred with the AI service: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    except Exception as e:
        # Catch any other unexpected errors
        print(f"🚨 [VIEW] An unexpected error occurred: {e}")
        return Response(
            {"error": f"An internal server error occurred: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )