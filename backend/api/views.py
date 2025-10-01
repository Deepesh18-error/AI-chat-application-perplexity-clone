from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Import our new services module and the custom exception
from . import services
from .services import GeminiError


@api_view(["POST"])
def generate_view(request):
    """
    API endpoint that receives a user prompt and returns a list of
    generated search queries.

    This view acts as an orchestrator:
    1. Validates the incoming prompt.
    2. Delegates the query generation task to the services layer.
    3. Handles potential errors from the service.
    4. Formats and returns the final response to the client.
    """
    try:
        prompt = request.data.get("prompt")

        if not prompt or not isinstance(prompt, str):
            return Response(
                {"error": "A valid 'prompt' string is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # --- THE DELEGATION STEP ---
        # Call our new specialist function from the services module.
        print(f"✅ [VIEW] Delegating prompt to services layer: '{prompt}'")
        queries = services.generate_search_queries(prompt)
        # ---------------------------

        # The new API "contract": return the list of queries.
        response_data = {"queries": queries}

        print(f"✅ [VIEW] Successfully received queries. Sending response to client.")
        return Response(response_data, status=status.HTTP_200_OK)

    except GeminiError as e:
        # --- CATCHING OUR CUSTOM ERROR ---
        # If the service layer raised a specific error,
        # we catch it here and return a user-friendly server error.
        print(f"🚨 [VIEW] A GeminiError occurred: {e}")
        return Response(
            {"error": f"An error occurred with the AI service: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    except Exception as e:
        # Catch any other unexpected errors that might occur in the view.
        print(f"🚨 [VIEW] An unexpected error occurred: {e}")
        return Response(
            {"error": "An internal server error occurred."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )