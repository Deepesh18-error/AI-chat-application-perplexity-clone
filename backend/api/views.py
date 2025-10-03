from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

# --- Our Application Imports ---
from . import services

@csrf_exempt  # Disable CSRF for our API. In production, you'd use a more secure method like token authentication.
@require_http_methods(["POST"]) # Enforce that this view only accepts POST requests.
async def generate_view(request):
    """
    The main API endpoint. It uses an intelligent router to decide the processing
    path and then streams structured Server-Sent Events (SSE) back to the client.
    
    This is a native Django async view and does NOT use the DRF @api_view decorator.
    """
    try:
        # 1. Manually parse the request body because we are not using DRF's `request.data`
        try:
            body = json.loads(request.body)
            prompt = body.get("prompt")
        except json.JSONDecodeError:
            # Use native Django JsonResponse for errors
            return JsonResponse(
                {"error": "Invalid JSON in request body."},
                status=400
            )

        if not prompt or not isinstance(prompt, str):
            return JsonResponse(
                {"error": "A valid 'prompt' string is required."},
                status=400
            )

        # 2. Decide the path (synchronous call)
        path = await services.decide_query_path_async(prompt)

        # 3. Get the main orchestrator/generator based on the path
        event_generator = services.generate_and_stream_answer(prompt, path)

        # 4. Wrap the generator in our SSE formatter
        sse_stream = services.stream_sse_formatter(event_generator)
        
        # 5. Return the native Django StreamingHttpResponse
        return StreamingHttpResponse(sse_stream, content_type="text/event-stream")

    except Exception as e:
        # This is a fallback for any unexpected errors during the setup phase
        print(f"🚨 [VIEW] A top-level error occurred in generate_view: {e}")
        return JsonResponse(
            {"error": f"An internal server error occurred: {e}"},
            status=500
        )