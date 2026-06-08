import json
import logging
import time
from datetime import datetime, timezone

from bson import ObjectId
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from . import services
from .auth import (
    auth_required,
    authenticate_credentials,
    create_access_token,
    create_refresh_token,
    create_user,
    public_user,
    refresh_user_access,
    revoke_user_refresh_tokens,
)
from .db_config import conversations_collection, ensure_database_indexes


logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
async def health_view(request):
    indexes_ready = await ensure_database_indexes()
    return JsonResponse({
        "status": "ok",
        "database": "connected" if conversations_collection is not None else "not_connected",
        "indexes": "ready" if indexes_ready else "not_ready",
    })


@csrf_exempt
@require_http_methods(["POST"])
async def register_view(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format in request body."}, status=400)

    email = (data.get("email") or "").strip()
    password = data.get("password") or ""
    name = (data.get("name") or "").strip()

    if not email or "@" not in email:
        return JsonResponse({"error": "A valid email is required."}, status=400)
    if len(password) < 8:
        return JsonResponse({"error": "Password must be at least 8 characters."}, status=400)

    user, error = await create_user(email=email, password=password, name=name)
    if error:
        status = 409 if "already exists" in error else 500
        return JsonResponse({"error": error}, status=status)

    token = create_access_token(user)
    return JsonResponse({
        "access_token": token,
        "refresh_token": create_refresh_token(user),
        "token_type": "Bearer",
        "user": public_user(user),
    }, status=201)


@csrf_exempt
@require_http_methods(["POST"])
async def login_view(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format in request body."}, status=400)

    email = data.get("email") or ""
    password = data.get("password") or ""

    user = await authenticate_credentials(email=email, password=password)
    if not user:
        return JsonResponse({"error": "Invalid email or password."}, status=401)

    token = create_access_token(user)
    return JsonResponse({
        "access_token": token,
        "refresh_token": create_refresh_token(user),
        "token_type": "Bearer",
        "user": public_user(user),
    })


@csrf_exempt
@require_http_methods(["POST"])
async def refresh_view(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format in request body."}, status=400)

    refresh_token = data.get("refresh_token") or ""
    access_token, new_refresh_token, error = await refresh_user_access(refresh_token)
    if error:
        status = 500 if error == "Database not connected" else 401
        return JsonResponse({"error": error}, status=status)

    return JsonResponse({
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "Bearer",
    })


@csrf_exempt
@require_http_methods(["POST"])
@auth_required
async def logout_view(request):
    revoked = await revoke_user_refresh_tokens(request.mongo_user_id)
    if not revoked:
        return JsonResponse({"error": "Could not revoke session."}, status=500)
    return JsonResponse({"status": "success"})


@require_http_methods(["GET"])
@auth_required
async def me_view(request):
    return JsonResponse({"user": public_user(request.mongo_user)})


@csrf_exempt
@require_http_methods(["POST"])
@auth_required
async def generate_view(request):
    logger.info("Received generation request.")
    try:
        server_entry_time = time.time() * 1000
        data = json.loads(request.body)

        prompt = data.get("prompt")
        session_id = data.get("session_id")
        context_package = data.get("context_package", {})
        force_web_search = data.get("force_web_search", False)
        ui_click_time = data.get("ui_click_time") or data.get("client_start_time")

        if not prompt or not session_id:
            error_msg = "Prompt and session_id are required."
            logger.warning("Generation request failed: %s", error_msg)
            return JsonResponse({"error": error_msg}, status=400)

        logger.info(
            "Generation prompt received. session_id=%s user_id=%s prompt_preview=%s",
            session_id,
            request.mongo_user_id,
            prompt[:100],
        )

        if ui_click_time:
            travel_time = server_entry_time - float(ui_click_time)
            logger.info("UI to server latency: %.2fms", travel_time)

        if conversations_collection is None:
            return JsonResponse({"error": "Database not connected"}, status=500)

        await ensure_database_indexes()

        previous_turns = await conversations_collection.find(
            {"user_id": request.mongo_user_id, "session_id": session_id}
        ).sort("turn_number", 1).to_list(length=50)

        turn_number = len(previous_turns) + 1
        context_package = {
            **(context_package if isinstance(context_package, dict) else {}),
            "current_query": prompt,
            "previous_turns": [
                {
                    "query": doc.get("user_query"),
                    "summary": doc.get("response_summary"),
                    "entities": doc.get("entities_mentioned", []),
                }
                for doc in previous_turns
                if doc.get("response_summary") != "Processing..."
            ],
        }

        if turn_number == 1:
            logger.info("Creating placeholder session record for new conversation.")
            initial_session_doc = {
                "user_id": request.mongo_user_id,
                "session_id": session_id,
                "turn_number": 1,
                "user_query": prompt,
                "response_summary": "Processing...",
                "entities_mentioned": [],
                "sources_used": [],
                "execution_path": "pending",
                "created_at": datetime.now(timezone.utc),
            }
            await conversations_collection.insert_one(initial_session_doc)
            logger.info("Placeholder session record created.")

        if force_web_search:
            path = "search_required"
            logger.info("Web search forced by the client.")
        else:
            logger.info("Running routing pipeline.")
            try:
                path = await services.get_intelligent_path(prompt, context_package)
            except services.ProviderError as exc:
                logger.warning("Routing provider fallback used: %s", exc.public_message)
                path = "direct_answer"

        event_generator = services.generate_and_stream_answer(
            prompt, path, session_id, turn_number, context_package, request.mongo_user_id
        )
        sse_stream = services.stream_sse_formatter(event_generator)

        response = StreamingHttpResponse(sse_stream, content_type="text/event-stream")
        response["X-Accel-Buffering"] = "no"
        response["Cache-Control"] = "no-cache"
        response["Connection"] = "keep-alive"

        logger.info("Streaming response started.")
        return response

    except json.JSONDecodeError:
        logger.warning("Generation request failed: invalid JSON.")
        return JsonResponse({"error": "Invalid JSON format in request body."}, status=400)
    except Exception as e:
        logger.exception("Unexpected generation error: %s", e)
        return JsonResponse({"error": "An internal server error occurred."}, status=500)


class ObjectIdEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)


@csrf_exempt
@require_http_methods(["GET"])
@auth_required
async def get_session_list(request):
    if conversations_collection is None:
        return JsonResponse({"error": "Database not connected"}, status=500)

    logger.info("Received request for session list.")

    try:
        await ensure_database_indexes()
        pipeline = [
            {"$match": {"user_id": request.mongo_user_id, "turn_number": 1}},
            {"$sort": {"created_at": -1}},
            {"$project": {"_id": 0, "session_id": 1, "title": "$chat_title"}},
        ]
        sessions = await conversations_collection.aggregate(pipeline).to_list(length=100)

        for session in sessions:
            title = session.get("title")
            if isinstance(title, str) and title.strip():
                session["title"] = title[:47] + "..." if len(title) > 50 else title
            else:
                session["title"] = "Untitled Chat"

        return JsonResponse(sessions, safe=False)

    except Exception as e:
        logger.exception("Session list error: %s", e)
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
@auth_required
async def get_session_history(request, session_id: str):
    if conversations_collection is None:
        return JsonResponse({"error": "Database not connected"}, status=500)

    try:
        await ensure_database_indexes()
        history_cursor = conversations_collection.find(
            {"user_id": request.mongo_user_id, "session_id": session_id}
        ).sort("turn_number", 1)
        history_docs = await history_cursor.to_list(length=None)

        formatted_history = []
        for doc in history_docs:
            final_aui_spec = f"<C1><P>{doc.get('response_summary', 'No summary available.')}</P></C1>"
            full_spec = doc.get("full_response_spec")
            if isinstance(full_spec, str) and full_spec.strip():
                final_aui_spec = full_spec

            formatted_history.append({
                "key": str(doc["_id"]),
                "prompt": doc.get("user_query"),
                "steps": ["Loaded from history"],
                "sources": doc.get("sources_used", []),
                "auiSpec": final_aui_spec,
                "streamingMarkdown": doc.get("full_markdown_response", ""),
                "error": None,
                "isLoading": False,
                "summary": doc.get("response_summary"),
                "entities": doc.get("entities_mentioned", []),
                "images": [],
                "isLoadedFromHistory": True,
            })
        return JsonResponse(formatted_history, safe=False, encoder=ObjectIdEncoder)
    except Exception as e:
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)


async def delete_session_view(request, session_id: str):
    logger.info("Deleting session: %s", session_id)
    try:
        if conversations_collection is None:
            return JsonResponse({"error": "Database not connected"}, status=500)

        await ensure_database_indexes()
        result = await conversations_collection.delete_many({
            "user_id": request.mongo_user_id,
            "session_id": session_id,
        })

        if result.deleted_count > 0:
            return JsonResponse({
                "status": "success",
                "message": f"Successfully deleted session {session_id}",
                "deleted_count": result.deleted_count,
            })

        return JsonResponse({
            "status": "not_found",
            "message": f"No session found with ID {session_id}",
        }, status=404)

    except Exception as e:
        logger.exception("Session deletion error: %s", e)
        return JsonResponse({"error": "An internal server error occurred during deletion."}, status=500)


@csrf_exempt
@require_http_methods(["GET", "DELETE"])
@auth_required
async def session_detail_view(request, session_id: str):
    if request.method == "GET":
        return await get_session_history(request, session_id)
    if request.method == "DELETE":
        return await delete_session_view(request, session_id)
    return JsonResponse({"error": "Method not allowed"}, status=405)
