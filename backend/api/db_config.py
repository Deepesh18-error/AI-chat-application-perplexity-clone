import os
import motor.motor_asyncio
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- DATABASE CONFIGURATION ---
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")

if not MONGO_CONNECTION_STRING:
    print("🚨 FATAL: MONGO_CONNECTION_STRING not found in environment variables.")
    db_client = None
    conversations_collection = None
else:
    try:
        print("✅ [DB_CONFIG] Connecting to local MongoDB...")
        db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_CONNECTION_STRING)
        
        # Define your database and collection names
        db = db_client.perplexity_clone_db
        conversations_collection = db.conversations
        
        print("✅ [DB_CONFIG] MongoDB client initialized successfully.")
        
    except Exception as e:
        print(f"🚨 [DB_CONFIG] Failed to connect to MongoDB: {e}")
        db_client = None
        conversations_collection = None