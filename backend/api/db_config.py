import os
import logging

import motor.motor_asyncio
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError


load_dotenv()

logger = logging.getLogger(__name__)

MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
db_client = None
users_collection = None
conversations_collection = None
_indexes_ready = False

if not MONGO_CONNECTION_STRING:
    logger.warning("MONGO_CONNECTION_STRING is not configured.")
else:
    try:
        logger.info("Initializing MongoDB client.")
        db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_CONNECTION_STRING)
        db = db_client.perplexity_clone_db
        users_collection = db.users
        conversations_collection = db.conversations
        logger.info("MongoDB client initialized.")
    except Exception as e:
        logger.exception("Failed to initialize MongoDB client: %s", e)


async def ensure_database_indexes() -> bool:
    global _indexes_ready

    if _indexes_ready:
        return True

    if users_collection is None or conversations_collection is None:
        return False

    try:
        await users_collection.create_index([("email", ASCENDING)], unique=True, name="users_email_unique")
    except PyMongoError as exc:
        logger.warning("Could not create unique user email index: %s", exc)

    index_specs = [
        (
            [("user_id", ASCENDING), ("session_id", ASCENDING), ("turn_number", ASCENDING)],
            "conversation_turn_lookup",
        ),
        (
            [("user_id", ASCENDING), ("created_at", DESCENDING)],
            "conversation_user_created_at",
        ),
        (
            [("user_id", ASCENDING), ("session_id", ASCENDING)],
            "conversation_session_lookup",
        ),
    ]

    for keys, name in index_specs:
        try:
            await conversations_collection.create_index(keys, name=name)
        except PyMongoError as exc:
            logger.warning("Could not create MongoDB index %s: %s", name, exc)

    _indexes_ready = True
    logger.info("MongoDB indexes are ready.")
    return True
