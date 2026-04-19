import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from datetime import datetime

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

try:
    cred = credentials.Certificate("service-account.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Firebase Error: {e}")

app = FastAPI(title="Bachatbot Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version='v1')
    )
except Exception as e:
    print(f"Gemini Connection Error: {e}")


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    user_id: str
    message: str                      # What the user typed in chat
    session_id: str | None = None     # Optional: for grouping a conversation


class SMSIncoming(BaseModel):
    user_id: str
    raw_sms: str
    sender: str                       # e.g. "eSewa", "NIC ASIA Bank"


class ConfirmAction(BaseModel):
    user_id: str
    confirmation_id: str
    confirmed: bool                   # True = Yes, False = No


# ---------------------------------------------------------------------------
# SYSTEM PROMPT  (the "brain" of the bot)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are Bachatbot, a friendly Nepali personal finance assistant. Think of yourself as a helpful saathi (friend) who helps users track money.

You understand Romanized Nepali (Nepali in English letters), mixed Nepali-English, and pure English.
Always reply in the SAME style/language the user uses. Keep replies short and warm — like texting a friend.

════════════════════════════════════════
EXPENSE CATEGORIES (ONLY these 7):
Food, Transport, Rent, Shopping, Health, Education, Others
════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — ONBOARDING (only for NEW users)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When a new user says hi/hello/namaste for the first time:

Step 1 — Introduce yourself warmly:
"Namaste! 👋 Ma Bachatbot ho — timro personal kharch tracker!
Ma timlai paisa manage garna help garxu. Suru garna ma kehi questions sodhxu, theek cha? 😊"

Step 2 — Ask about their income first:
"Pahile bata — timro monthly income kati ho? (Salary, freelance, business — jे भए पनि)"

Step 3 — After income, ask about regular expenses ONE BY ONE:
Ask each category separately in a natural flowing way:
- "Rent/kotha ma monthly kati tirxau?"
- "Khana-pina ma (groceries + baira khana) roughly kati jaanxa?"
- "Transport ma (bus, fuel, taxi) kati lagxa?"
- "Aaru regular kharcha xa? (health, education, shopping, etc.)"

Step 4 — Summarize everything and ask for confirmation:
Show a clean summary like:
"Okay! Yesto bhayo timro monthly breakdown:
💰 Income: Rs.X
🏠 Rent: Rs.X
🍛 Food: Rs.X
🚌 Transport: Rs.X
...
Yei sab add garau? 'hao' bhanyo bhaney sab save hunxa! ✅"

Step 5 — When user confirms with hao/yes/okay/ha:
Emit ONE <ACTION> block per item. Emit ALL of them together in one reply.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — DAILY TRACKING (returning users)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPENSE DETECTION:
- "khana ma 500 gayo", "momo khayen 200", "bus ma 50 cha"
- Extract item + amount + category
- Confirm before saving:
  "Food ma Rs.500 add garau? (khana) — 'hao' bhanyo bhaney hunxa!"

INCOME DETECTION:
- "salary aayo 45000", "eSewa bata paisa aayo", "freelance ko 8000 aayo"
- Confirm: "Rs.45000 income ma add garau? — 'hao' bhanyo bhaney hunxa 💰"

ON CONFIRMATION (hao / yes / okay / ha / gara / add gara):
- Emit the ACTION block(s) for what was just discussed.

ON CANCELLATION (nai / no / nakhanu / chhadau):
- "Theek cha, add garindaina! Aaru kei xa?" and move on.

GENERAL QUESTIONS:
- "balance kati xa?" → "Dashboard ma check gara! Ma tracking gariraxa 📊"
- "rent ma kati gayo?" → Answer based on conversation context if available.
- "help" or confused user → Explain what you can do simply.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTION BLOCK FORMAT (CRITICAL RULES)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Only emit when user has CONFIRMED. Never speculatively.
Always put friendly text BEFORE the action block(s).

For expense:
<ACTION>{"type":"expense","item":"khana","amount":500,"category":"Food"}</ACTION>

For income:
<ACTION>{"type":"income","source":"salary","amount":45000}</ACTION>

For MULTIPLE items (onboarding confirmation or user shares multiple at once):
Emit multiple ACTION blocks one after another:
<ACTION>{"type":"income","source":"salary","amount":45000}</ACTION>
<ACTION>{"type":"expense","item":"rent","amount":14000,"category":"Rent"}</ACTION>
<ACTION>{"type":"expense","item":"khana","amount":3000,"category":"Food"}</ACTION>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL ONBOARDING EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User: "hi"
Bot: "Namaste! 👋 Ma Bachatbot ho — timro personal kharch tracker! Ma timlai paisa manage garna help garxu. Suru garna kehi questions sodhxu, theek cha? 😊"

User: "hao theek xa"
Bot: "Ramro! Pahile bata — timro monthly income kati ho? (salary, freelance, business — je bhaye pani)"

User: "45000 salary aauxa"
Bot: "Nice! Rs.45000 income. Ab bata — rent/kotha ma monthly kati tirxau?"

User: "14000"
Bot: "Okay! Rent Rs.14000. Khana-pina ma (groceries + baira khana) roughly kati jaanxa mahina ma?"

User: "3000 tira jaanxa"
Bot: "Theek xa! Transport ma (bus, fuel, taxi) kati lagxa?"

User: "1000 lagxa"
Bot: "Ramro! Aaru kei regular kharcha xa? (health, education, shopping) — xa bhane bata, chha bhane directly summary herna sakxau."

User: "chha"
Bot: "Okay! Yesto bhayo timro monthly breakdown:
💰 Income: Rs.45,000
🏠 Rent: Rs.14,000
🍛 Food: Rs.3,000
🚌 Transport: Rs.1,000
📊 Balance: Rs.27,000

Yei sab add garau? 'hao' bhanyo bhaney sab save hunxa! ✅"

User: "hao"
Bot: "Perfect! Sab save gariyako cha ✅ Ab daily kharcha hunxa bhaney malai bata — ma track garirahanxu! 💪
<ACTION>{"type":"income","source":"salary","amount":45000}</ACTION>
<ACTION>{"type":"expense","item":"rent","amount":14000,"category":"Rent"}</ACTION>
<ACTION>{"type":"expense","item":"khana","amount":3000,"category":"Food"}</ACTION>
<ACTION>{"type":"expense","item":"transport","amount":1000,"category":"Transport"}</ACTION>"
"""


# ---------------------------------------------------------------------------
# FIRESTORE HELPERS
# ---------------------------------------------------------------------------

def _month_key() -> str:
    return datetime.utcnow().strftime("%Y-%m")


def _save_expense(user_id: str, item: str, amount: int, category: str):
    """Write expense doc + update monthly_summary atomically."""
    month = _month_key()
    summary_ref = (
        db.collection("users").document(user_id)
          .collection("monthly_summary").document(month)
    )
    expense_ref = (
        db.collection("users").document(user_id)
          .collection("expenses").document()
    )

    @firestore.transactional
    def _txn(transaction, summary_ref, expense_ref):
        snap = summary_ref.get(transaction=transaction)
        if not snap.exists:
            transaction.set(summary_ref, {
                "income_total": 0, "expense_total": 0, "balance": 0,
                "categories": {c: 0 for c in
                    ["Food", "Transport", "Rent", "Shopping",
                     "Health", "Education", "Others"]},
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
            cats = {}
            current_expense = 0
            current_income = 0
        else:
            d = snap.to_dict()
            cats = d.get("categories", {})
            current_expense = d.get("expense_total", 0)
            current_income = d.get("income_total", 0)

        cats[category] = cats.get(category, 0) + amount
        new_expense = current_expense + amount
        new_balance = current_income - new_expense

        transaction.set(expense_ref, {
            "item": item, "amount": amount, "category": category,
            "month": month, "timestamp": firestore.SERVER_TIMESTAMP,
        })
        transaction.update(summary_ref, {
            "expense_total": new_expense,
            "balance": new_balance,
            "categories": cats,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

    t = db.transaction()
    _txn(t, summary_ref, expense_ref)


def _save_income(user_id: str, source: str, amount: int):
    """Write income doc + update monthly_summary atomically."""
    month = _month_key()
    summary_ref = (
        db.collection("users").document(user_id)
          .collection("monthly_summary").document(month)
    )
    income_ref = (
        db.collection("users").document(user_id)
          .collection("income").document()
    )

    @firestore.transactional
    def _txn(transaction, summary_ref, income_ref):
        snap = summary_ref.get(transaction=transaction)
        if not snap.exists:
            transaction.set(summary_ref, {
                "income_total": 0, "expense_total": 0, "balance": 0,
                "categories": {c: 0 for c in
                    ["Food", "Transport", "Rent", "Shopping",
                     "Health", "Education", "Others"]},
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
            current_expense = 0
            current_income = 0
        else:
            d = snap.to_dict()
            current_expense = d.get("expense_total", 0)
            current_income = d.get("income_total", 0)

        new_income = current_income + amount
        new_balance = new_income - current_expense

        transaction.set(income_ref, {
            "source": source, "amount": amount,
            "month": month, "timestamp": firestore.SERVER_TIMESTAMP,
        })
        transaction.update(summary_ref, {
            "income_total": new_income,
            "balance": new_balance,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

    t = db.transaction()
    _txn(t, summary_ref, income_ref)


def _get_history(user_id: str, limit: int = 10) -> list[dict]:
    """Load last N chat messages for this user from Firestore."""
    docs = (
        db.collection("users").document(user_id)
          .collection("chat_history")
          .order_by("timestamp", direction=firestore.Query.DESCENDING)
          .limit(limit)
          .stream()
    )
    messages = [d.to_dict() for d in docs]
    messages.reverse()  # oldest first for Gemini context
    return messages


def _save_message(user_id: str, role: str, text: str):
    """Persist a single chat turn to Firestore."""
    db.collection("users").document(user_id)\
      .collection("chat_history").add({
          "role": role,          # "user" or "model"
          "text": text,
          "timestamp": firestore.SERVER_TIMESTAMP,
      })


def _parse_action(bot_reply: str) -> list[dict]:
    """Extract ALL <ACTION>{...}</ACTION> blocks — returns a list.
    Handles single expense AND onboarding multi-save in one reply."""
    import re
    matches = re.findall(r"<ACTION>(.*?)</ACTION>", bot_reply, re.DOTALL)
    actions = []
    for m in matches:
        try:
            actions.append(json.loads(m.strip()))
        except json.JSONDecodeError:
            continue
    return actions


def _strip_action(bot_reply: str) -> str:
    """Remove the ACTION block from the text shown to the user."""
    import re
    return re.sub(r"<ACTION>.*?</ACTION>", "", bot_reply, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# CHAT ENDPOINT  — main conversational endpoint
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(payload: ChatMessage):
    """
    Send a user message → get bot reply.
    If the bot detects a confirmed action, it saves to Firestore automatically.

    Response shape:
    {
        "reply": "Bot's friendly message (Romanized Nepali)",
        "action_taken": null | {"type": "expense"|"income", ...},
        "saved": true | false
    }
    """
    user_id = payload.user_id
    user_text = payload.message.strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message")

    # 1. Load recent conversation history
    history = _get_history(user_id, limit=10)

    # 2. Build Gemini contents list (system + history + new message)
    contents = []

    # Inject history as alternating user/model turns
    for turn in history:
        contents.append(
            types.Content(
                role=turn["role"],
                parts=[types.Part(text=turn["text"])]
            )
        )

    # Append current user message
    contents.append(
        types.Content(role="user", parts=[types.Part(text=user_text)])
    )

    # 3. Call Gemini with system prompt
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,   # low = more consistent, predictable actions
                max_output_tokens=300,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

    bot_reply_raw = response.text or ""

    # 4. Parse any ACTION block(s) — can be multiple on onboarding confirmation
    actions = _parse_action(bot_reply_raw)
    bot_reply_clean = _strip_action(bot_reply_raw)

    # 5. Execute all actions
    saved_actions = []
    for action in actions:
        try:
            if action.get("type") == "expense":
                _save_expense(
                    user_id=user_id,
                    item=action.get("item", "unknown"),
                    amount=int(action.get("amount", 0)),
                    category=action.get("category", "Others"),
                )
                saved_actions.append(action)
            elif action.get("type") == "income":
                _save_income(
                    user_id=user_id,
                    source=action.get("source", "unknown"),
                    amount=int(action.get("amount", 0)),
                )
                saved_actions.append(action)
        except Exception as e:
            bot_reply_clean += f"\n(⚠️ Save error: {str(e)})"

    saved = len(saved_actions) > 0

    # 6. Persist both turns to Firestore for future context
    _save_message(user_id, "user", user_text)
    _save_message(user_id, "model", bot_reply_raw)  # save raw (with ACTION) for context

    return {
        "reply": bot_reply_clean,
        "actions_taken": saved_actions,   # list — empty [] if nothing saved
        "saved": saved,
    }


# ---------------------------------------------------------------------------
# SMS DETECTION ENDPOINT  — called by Flutter when an SMS arrives
# ---------------------------------------------------------------------------

SMS_PARSE_PROMPT = """
You are a financial SMS parser. Analyze the SMS below.
Only process messages from banks, eSewa, Khalti, IME Pay, ConnectIPS.
Ignore OTPs, promotional messages, social media notifications.

Return ONLY raw JSON (no markdown, no explanation):
{
  "is_financial": true or false,
  "type": "income" or "expense",
  "amount": <integer> or null,
  "source": "<sender name>",
  "description": "<one short phrase>"
}
"""

@app.post("/sms-detect")
async def sms_detect(payload: SMSIncoming):
    """
    Flutter sends raw SMS text here.
    Bot parses it and writes a pending_confirmation doc.
    Flutter reads that doc to show the confirm dialog.

    Response:
    {
        "is_financial": bool,
        "bot_message": "Timile 500 eSewa bata ayo...",
        "confirmation_id": "<firestore doc id>"
    }
    """
    sms_text = f"Sender: {payload.sender}\nMessage: {payload.raw_sms}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=sms_text,
            config=types.GenerateContentConfig(
                system_instruction=SMS_PARSE_PROMPT,
                temperature=0.1,
                max_output_tokens=150,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

    clean = response.text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return {"is_financial": False, "bot_message": None, "confirmation_id": None}

    if not parsed.get("is_financial") or not parsed.get("amount"):
        return {"is_financial": False, "bot_message": None, "confirmation_id": None}

    # Build the Nepali confirmation message for the chatbot UI
    amt = parsed["amount"]
    src = parsed["source"]
    typ = parsed["type"]

    if typ == "income":
        bot_msg = f"Timile Rs.{amt} '{src}' bata payau jasto lagcha! 💸 Yo income ma add gardeu? 'hao' bhanyo bhaney huncha."
    else:
        bot_msg = f"Rs.{amt} '{src}' ma gayo jasto lagcha. Yo expense ma add gardeu? 'hao' bhanyo bhaney huncha."

    # Write pending confirmation to Firestore
    ref = (
        db.collection("users").document(payload.user_id)
          .collection("pending_confirmations").document()
    )
    ref.set({
        "raw_sms": payload.raw_sms,
        "sender": payload.sender,
        "parsed": parsed,
        "bot_message": bot_msg,
        "status": "pending",
        "created_at": firestore.SERVER_TIMESTAMP,
    })

    return {
        "is_financial": True,
        "bot_message": bot_msg,
        "confirmation_id": ref.id,
    }


# ---------------------------------------------------------------------------
# SMS CONFIRMATION ENDPOINT  — called when user taps Yes/No in Flutter
# ---------------------------------------------------------------------------

@app.post("/sms-confirm")
async def sms_confirm(payload: ConfirmAction):
    """
    User tapped Yes/No on the SMS confirmation dialog.
    If Yes → saves income/expense to Firestore.

    Response:
    {
        "status": "confirmed" | "rejected",
        "saved": bool
    }
    """
    ref = (
        db.collection("users").document(payload.user_id)
          .collection("pending_confirmations").document(payload.confirmation_id)
    )
    doc = ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Confirmation not found")

    data = doc.to_dict()
    if data.get("status") != "pending":
        raise HTTPException(status_code=409, detail="Already handled")

    if not payload.confirmed:
        ref.update({"status": "rejected"})
        return {"status": "rejected", "saved": False}

    parsed = data["parsed"]
    saved = False

    try:
        if parsed["type"] == "income":
            _save_income(
                user_id=payload.user_id,
                source=parsed.get("source", "unknown"),
                amount=int(parsed["amount"]),
            )
            saved = True
        else:
            _save_expense(
                user_id=payload.user_id,
                item=parsed.get("description", "sms expense"),
                amount=int(parsed["amount"]),
                category="Others",
            )
            saved = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save error: {str(e)}")

    ref.update({"status": "confirmed"})
    return {"status": "confirmed", "saved": saved}


# ---------------------------------------------------------------------------
# CLEAR HISTORY  — useful for testing / "new conversation" button
# ---------------------------------------------------------------------------

@app.delete("/chat/history/{user_id}")
async def clear_history(user_id: str):
    """Delete all chat history for a user (fresh start)."""
    col = db.collection("users").document(user_id).collection("chat_history")
    docs = col.stream()
    for d in docs:
        d.reference.delete()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "Bachatbot",
        "status": "running ✅",
        "model": "gemini-2.5-flash-lite",
        "endpoints": ["/chat", "/sms-detect", "/sms-confirm"],
    }