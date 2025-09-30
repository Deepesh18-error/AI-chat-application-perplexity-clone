from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(["POST"])
def generate_view(request):
    """
    API endpoint to receive a user prompt.

    This view is designed to be the primary entry point for the query
    processing pipeline. In Stage 1, its only responsibility is to
    confirm that it has received the prompt from the frontend.

    Args:
        request: The Django HttpRequest object. The user's prompt is
                 expected in the request body as JSON: {"prompt": "..."}.

    Returns:
        Response: A DRF Response object.
                  - 200 OK: If the prompt is received successfully.
                  - 400 Bad Request: If the 'prompt' key is missing or empty.
    """
    try:
        prompt = request.data.get("prompt")

        if not prompt or not isinstance(prompt, str):
            # If the prompt is missing or not a string, return an error
            return Response(
                {"error": "A valid 'prompt' string is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        print(f"✅ [API] Received prompt: '{prompt}'")

        response_data = {"message": f"Prompt received successfully."}

        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        # Generic error handler 
        print(f"🚨 [API] An unexpected error occurred: {e}")
        return Response(
            {"error": "An internal server error occurred."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )