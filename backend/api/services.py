# In backend/api/services.py

import os  # <-- Import the os module directly
import json
import google.generativeai as genai
from typing import List
# We no longer need to import from django.conf

# --- ROBUST CONFIGURATION AT MODULE LEVEL ---
# Read the API key directly from the environment variable.
# This is the most direct and reliable way.
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    # Configure the library ONCE when the module is loaded.
    genai.configure(api_key=api_key)
    print("✅ [SERVICES] Google Generative AI configured successfully.")
except Exception as e:
    print(f"🚨 [SERVICES] Failed to configure Google Generative AI: {e}")
# ---------------------------------------------


class GeminiError(Exception):
    """Custom exception for errors related to the Gemini API."""
    pass


def generate_search_queries(prompt: str) -> List[str]:
    """
    Takes a user's natural language prompt and uses the Gemini API
    to generate a list of targeted search queries.
    """
    try:
        # --- Define the Generation Configuration ---
        generation_config = {
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
        }

        # --- Select the Model (Use the correct, stable model name) ---
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",  # <-- Using the stable "latest" tag
            generation_config=generation_config,
        )

        # --- Craft the Prompt ---
        system_prompt = """
        You are an expert search query generation assistant. Your sole purpose is to
        analyze the user's question and decompose it into 3 to 5 simple, effective
        search engine queries that will gather the necessary information to provide a
        comprehensive answer.

        Do not answer the question yourself. Do not add any commentary or explanation.

        You must respond with ONLY a valid JSON object. The JSON object must have a
        single key named "queries", which contains a list of the generated search
        query strings.
        """
        full_prompt = [system_prompt, "user: ", prompt]

        # --- Execute the API Call ---
        print(f"✅ [SERVICES] Sending prompt to Gemini: '{prompt}'")
        response = model.generate_content(full_prompt)

        # --- Parse and Validate the Response ---
        response_json = json.loads(response.text)
        
        if "queries" not in response_json or not isinstance(response_json["queries"], list):
            raise GeminiError("Invalid JSON response format from Gemini: 'queries' key is missing or not a list.")
            
        queries = response_json["queries"]
        
        if not all(isinstance(q, str) for q in queries):
            raise GeminiError("Invalid JSON response format from Gemini: not all items in 'queries' are strings.")

        print(f"✅ [SERVICES] Successfully received and parsed queries from Gemini: {queries}")
        return queries

    except Exception as e:
        print(f"🚨 [SERVICES] An error occurred while calling the Gemini API: {e}")
        raise GeminiError(f"Failed to generate search queries: {e}")