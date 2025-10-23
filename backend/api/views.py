# backend/api/views.py - FINAL CORRECTED VERSION

from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from bson import ObjectId
from datetime import datetime, timezone
from .db_config import conversations_collection

# --- THE CRITICAL FIX: IMPORT THE 'services' MODULE ---
from . import services
# --------------------------------------------------------

from .db_config import conversations_collection

@csrf_exempt
@require_http_methods(["POST"])
async def generate_view(request):
    print("\n--- [VIEW] Received new generation request ---")
    try:
        data = json.loads(request.body)
        prompt = data.get("prompt")
        session_id = data.get("session_id")
        turn_number = data.get("turn_number")
        context_package = data.get("context_package", {})

        if not prompt or not session_id or not turn_number:
            error_msg = "Prompt, session_id, and turn_number are required."
            print(f"🚨 [VIEW] Request failed: {error_msg}")
            return JsonResponse({"error": error_msg}, status=400)

        print(f"✅ [VIEW] Prompt received: '{prompt[:100]}...'")
        print(f"  > Session ID: {session_id}, Turn: {turn_number}")
        
        if turn_number == 1:
            print(f"  > First turn detected. Pre-creating session record in DB.")
            initial_session_doc = {
                "session_id": session_id,
                "turn_number": 1,
                "user_query": prompt,
                "response_summary": "Processing...",  # Placeholder summary
                "entities_mentioned": [],
                "sources_used": [],
                "execution_path": "pending",
                "created_at": datetime.now(timezone.utc)
            }
            # This is a blocking call - we wait for it to finish
            await conversations_collection.insert_one(initial_session_doc)
            print("  > Initial record created successfully.")

        path = await services.get_intelligent_path(prompt, context_package)
        
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
        import traceback
        traceback.print_exc() # Print full traceback to the console for debugging
        return JsonResponse({"error": "An internal server error occurred."}, status=500)

# --- NEW CODE FOR CHAT HISTORY ---

class ObjectIdEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super(ObjectIdEncoder, self).default(obj)

@csrf_exempt
@require_http_methods(["GET"])
async def get_session_list(request):
    if conversations_collection is None:
        return JsonResponse({"error": "Database not connected"}, status=500)
    
    print("✅ [VIEW] Received request for session list.")
    
    try:
        # Step 1: Fetch the initial data from MongoDB. This part of your code was correct.
        pipeline = [
            {"$match": {"turn_number": 1}},
            {"$sort": {"created_at": -1}},
            {"$project": {"_id": 0, "session_id": 1, "title": "$chat_title"}} # <-- CHANGED THIS LINE
        ]
        sessions = await conversations_collection.aggregate(pipeline).to_list(length=100)
        
        print(f"  > Found {len(sessions)} sessions from database.")

        # Step 2: Safely process the titles. This is the corrected logic.
        for session in sessions:
            # Check if 'title' exists, is a string, and is not just empty spaces.
            if 'title' in session and isinstance(session.get('title'), str) and session['title'].strip():
                # If the title is valid and too long, truncate it.
                if len(session['title']) > 50:
                    session['title'] = session['title'][:47] + "..."
                # If the title is valid and short, we do nothing and leave it as is.
            else:
                # If the title is missing or invalid, provide a safe, user-friendly fallback.
                session['title'] = "Untitled Chat"
        
        print("  > Successfully processed titles.")
        return JsonResponse(sessions, safe=False)

    except Exception as e:
        print(f"🚨 [VIEW] An error occurred in get_session_list: {str(e)}")
        # This will catch any other unexpected errors and return a proper error response.
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)

@csrf_exempt
@require_http_methods(["GET"])
async def get_session_history(request, session_id: str):
    if conversations_collection is None:
        return JsonResponse({"error": "Database not connected"}, status=500)
    try:
        # It finds all turns for the session
        history_cursor = conversations_collection.find(
            {'session_id': session_id}
        ).sort('turn_number', 1) # It sorts them correctly
        history_docs = await history_cursor.to_list(length=None)
        
        # It formats the data for the frontend
        formatted_history = []
        for doc in history_docs:
            
            # --- START OF MODIFICATION ---
            
            # Default to a simple summary-based spec
            final_aui_spec = f"<C1><P>{doc.get('response_summary', 'No summary available.')}</P></C1>"
            
            # Check if the high-fidelity spec exists and has content
            full_spec = doc.get("full_response_spec")
            if full_spec and isinstance(full_spec, str) and full_spec.strip():
                # If it does, use it instead of the summary
                final_aui_spec = full_spec
                
            # --- END OF MODIFICATION ---

            formatted_history.append({
                "key": str(doc['_id']),
                "prompt": doc.get('user_query'),
                "steps": ["Loaded from history"],
                "sources": doc.get('sources_used', []),
                "auiSpec": final_aui_spec, # <-- USE THE FINAL SPEC
                "error": None,
                "isLoading": False,
                "summary": doc.get('response_summary'),
                "entities": doc.get('entities_mentioned', []),
                "images": [],
                "isLoadedFromHistory": True
            })
        return JsonResponse(formatted_history, safe=False, encoder=ObjectIdEncoder)
    except Exception as e:
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)