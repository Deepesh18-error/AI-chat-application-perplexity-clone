# api/views.py

from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from . import services

@csrf_exempt
@require_http_methods(["POST"])
async def generate_view(request):
    """(FINAL CORRECTED VERSION)
    Receives session/context and passes it correctly to the services layer.
    """
    print("\n--- [VIEW] Received new generation request ---")
    try:
        data = json.loads(request.body)
        prompt = data.get("prompt")
        # --- RECEIVE ALL DATA FROM FRONTEND ---
        session_id = data.get("session_id")
        turn_number = data.get("turn_number")
        context_package = data.get("context_package", {})

        if not prompt or not session_id or not turn_number:
            error_msg = "Prompt, session_id, and turn_number are required."
            print(f"🚨 [VIEW] Request failed: {error_msg}")
            return JsonResponse({"error": error_msg}, status=400)

        print(f"✅ [VIEW] Prompt received: '{prompt[:100]}...'")
        print(f"  > Session ID: {session_id}, Turn: {turn_number}")
        
        # --- FIX: Call get_intelligent_path with the correct arguments ---
        path = await services.get_intelligent_path(prompt, context_package)
        
        # --- FIX: Call generate_and_stream_answer with ALL arguments ---
        event_generator = services.generate_and_stream_answer(
            prompt, path, session_id, turn_number, context_package
        )
        
        sse_stream = services.stream_sse_formatter(event_generator)
        
        response = StreamingHttpResponse(sse_stream, content_type='text/event-stream')
        response['X-Accel-Buffering'] = 'no'
        response['Cache-Control'] = 'no-cache'
        
        print("✅ [VIEW] Streaming response started.")
        return response

    except json.JSONDecodeError:
        print("🚨 [VIEW] Request failed: Invalid JSON in request body.")
        return JsonResponse({"error": "Invalid JSON format in request body."}, status=400)
    except Exception as e:
        print(f"🚨 [VIEW] An unexpected error occurred in the view: {e}")
        return JsonResponse({"error": "An internal server error occurred."}, status=500)