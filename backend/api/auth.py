import jwt
from datetime import datetime, timedelta, timezone
from functools import wraps

from bson import ObjectId
from django.conf import settings
from django.contrib.auth.hashers import check_password, make_password
from django.http import JsonResponse
from pymongo.errors import DuplicateKeyError

from .db_config import ensure_database_indexes, users_collection


ACCESS_TOKEN_MINUTES = int(getattr(settings, "JWT_ACCESS_TOKEN_MINUTES", 60))
REFRESH_TOKEN_DAYS = int(getattr(settings, "JWT_REFRESH_TOKEN_DAYS", 14))


def normalize_email(email: str) -> str:
    return email.strip().lower()


def public_user(user: dict) -> dict:
    created_at = user.get("created_at")
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "name": user.get("name", ""),
        "created_at": created_at.isoformat() if created_at else None,
    }


def create_access_token(user: dict) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["_id"]),
        "email": user["email"],
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_MINUTES)).timestamp()),
        "type": "access",
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


def create_refresh_token(user: dict) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["_id"]),
        "email": user["email"],
        "token_version": user.get("token_version", 0),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=REFRESH_TOKEN_DAYS)).timestamp()),
        "type": "refresh",
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


def decode_access_token(token: str) -> dict | None:
    payload = decode_token(token)
    if not payload:
        return None

    if payload.get("type") != "access":
        return None
    return payload


def decode_refresh_token(token: str) -> dict | None:
    payload = decode_token(token)
    if not payload:
        return None

    if payload.get("type") != "refresh":
        return None
    return payload


async def create_user(email: str, password: str, name: str = "") -> tuple[dict | None, str | None]:
    if users_collection is None:
        return None, "Database not connected"

    clean_email = normalize_email(email)
    await ensure_database_indexes()

    existing_user = await users_collection.find_one({"email": clean_email})
    if existing_user:
        return None, "An account with this email already exists."

    now = datetime.now(timezone.utc)
    user_doc = {
        "email": clean_email,
        "name": name.strip(),
        "password_hash": make_password(password),
        "created_at": now,
        "updated_at": now,
        "last_login": None,
        "token_version": 0,
    }
    try:
        result = await users_collection.insert_one(user_doc)
    except DuplicateKeyError:
        return None, "An account with this email already exists."

    user_doc["_id"] = result.inserted_id
    return user_doc, None


async def authenticate_credentials(email: str, password: str) -> dict | None:
    if users_collection is None:
        return None

    user = await users_collection.find_one({"email": normalize_email(email)})
    if not user:
        return None

    if not check_password(password, user.get("password_hash", "")):
        return None

    await users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.now(timezone.utc)}},
    )
    return user


async def refresh_user_access(refresh_token: str) -> tuple[dict | None, str | None, str | None]:
    payload = decode_refresh_token(refresh_token)
    if not payload:
        return None, None, "Invalid refresh token."

    try:
        user_id = ObjectId(payload["sub"])
    except Exception:
        return None, None, "Invalid refresh token."

    if users_collection is None:
        return None, None, "Database not connected"

    user = await users_collection.find_one({"_id": user_id})
    if not user:
        return None, None, "User not found."

    if payload.get("token_version") != user.get("token_version", 0):
        return None, None, "Refresh token has been revoked."

    return create_access_token(user), create_refresh_token(user), None


async def revoke_user_refresh_tokens(user_id: str) -> bool:
    if users_collection is None:
        return False

    try:
        object_id = ObjectId(user_id)
    except Exception:
        return False

    result = await users_collection.update_one(
        {"_id": object_id},
        {"$inc": {"token_version": 1}, "$set": {"updated_at": datetime.now(timezone.utc)}},
    )
    return result.modified_count > 0


async def get_user_from_request(request) -> dict | None:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header.removeprefix("Bearer ").strip()
    payload = decode_access_token(token)
    if not payload:
        return None

    try:
        user_id = ObjectId(payload["sub"])
    except Exception:
        return None

    if users_collection is None:
        return None

    return await users_collection.find_one({"_id": user_id})


def auth_required(view_func):
    @wraps(view_func)
    async def wrapper(request, *args, **kwargs):
        user = await get_user_from_request(request)
        if not user:
            return JsonResponse({"error": "Authentication required."}, status=401)

        request.mongo_user = user
        request.mongo_user_id = str(user["_id"])
        return await view_func(request, *args, **kwargs)

    return wrapper
