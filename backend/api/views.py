# api/views.py

from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from . import services

@csrf_exempt
@require_http_methods(["POST"])
async def generate_view(request):
    """
    The main API endpoint for the Perplexity Clone.
    It receives a prompt, determines the correct execution path,
    and streams back the results as Server-Sent Events (SSE).
    """
    print("\n--- [VIEW] Received new generation request ---")
    try:
        data = json.loads(request.body)
        prompt = data.get("prompt")

        if not prompt:
            print("🚨 [VIEW] Request failed: No prompt provided.")
            return JsonResponse({"error": "Prompt is required in the request body."}, status=400)

        print(f"✅ [VIEW] Prompt received: '{prompt[:100]}...'")
        
        # 1. Use the Intelligent Classifier to decide the path
        path = await services.get_intelligent_path(prompt)
        
        # 2. Get the main event generator from the orchestrator
        event_generator = services.generate_and_stream_answer(prompt, path)
        
        # 3. Wrap the generator with the SSE formatter
        sse_stream = services.stream_sse_formatter(event_generator)
        
        # 4. Return the stream to the client
        # The 'text/event-stream' content type is essential for SSEs to work.
        response = StreamingHttpResponse(sse_stream, content_type='text/event-stream')
        response['X-Accel-Buffering'] = 'no'  # Disable buffering for Nginx
        response['Cache-Control'] = 'no-cache' # Ensure client doesn't cache the stream
        
        print("✅ [VIEW] Streaming response started.")
        return response

    except json.JSONDecodeError:
        print("🚨 [VIEW] Request failed: Invalid JSON in request body.")
        return JsonResponse({"error": "Invalid JSON format in request body."}, status=400)
    except Exception as e:
        print(f"🚨 [VIEW] An unexpected error occurred in the view: {e}")
        return JsonResponse({"error": "An internal server error occurred."}, status=500)