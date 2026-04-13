import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load your .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# --- FIREBASE INITIALIZATION ---
# --- FIREBASE INITIALIZATION ---
# Use the file you downloaded from Firebase Project Settings
try:
    cred = credentials.Certificate("service-account.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Firebase Error: {e}")

# 2. Initialize the FastAPI app
app = FastAPI(title="Bachatbot AI Backend")

# 3. Create the Gemini Client
try:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version='v1')
    )
except Exception as e:
    print(f"Connection Error: {e}")

@app.get("/")
def home():
    status = "Connected ✅" if api_key else "Missing Key ❌"
    return {
        "project": "Bachatbot",
        "api_status": status,
        "model": "Gemini 2.5 Flash Lite"
    }

@app.post("/parse")
async def parse_expense(user_input: str, user_id: str = "default_user"):
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured in .env")

    ALLOWED_CATEGORIES = "Food, Transport, Rent, Shopping, Health, Education, Others"

    system_instruction = (
        f"You are a Nepali expense tracker. Your goal is to identify expenses from the text.\n"
        f"1. Categorize each item into ONLY one of these: [{ALLOWED_CATEGORIES}].\n"
        f"2. Convert Nepali words to their English category (e.g., 'khana' -> 'Food', 'bus' -> 'Transport').\n"
        f"3. Return ONLY a JSON object with this structure: "
        "{'expenses': [{'item': string, 'amount': integer, 'category': string}]}"
    )

    try:
        # Step 1: Get AI Analysis
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=f"{system_instruction}\nInput: {user_input}"
        )

        if not response.text:
            raise HTTPException(status_code=500, detail="AI returned an empty response")

        # Clean JSON string (removes markdown ```json ... ```)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        analysis_json = json.loads(clean_text)

        # Step 2: SAVE TO FIREBASE
        doc_ref = db.collection("expenses").document()
        doc_ref.set({
            "user_id": user_id,
            "raw_text": user_input,
            "structured_data": analysis_json,
            "timestamp": firestore.SERVER_TIMESTAMP
        })

        return {
            "status": "Success & Saved ✅",
            "firebase_id": doc_ref.id,
            "analysis": analysis_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))