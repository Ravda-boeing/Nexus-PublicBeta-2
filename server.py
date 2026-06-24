import os
import sqlite3
import json
import uuid
import traceback
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

# ─── Gemini SDK Setup ────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash-lite"

# ─── Database ─────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("nexus.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at TEXT,
        messages TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

# ─── In-memory sessions ───────────────────────────────────────────
active_sessions: dict = {}

# ─── Models ───────────────────────────────────────────────────────
class MessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# ─── Routes ───────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: MessageRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())

        # Load from DB into memory if needed
        if session_id not in active_sessions:
            conn = sqlite3.connect("nexus.db")
            c = conn.cursor()
            c.execute("SELECT messages FROM conversations WHERE id = ?", (session_id,))
            row = c.fetchone()
            conn.close()
            active_sessions[session_id] = json.loads(row[0]) if row else []

        # Add user message
        active_sessions[session_id].append({
            "role": "user",
            "content": request.message
        })

        # Convert to new SDK format
        history = []
        messages = active_sessions[session_id]
        for msg in messages[:-1]:  # all but the latest user message
            role = "model" if msg["role"] == "assistant" else "user"
            history.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])]
            ))

        # Send with history
        response = client.models.generate_content(
            model=MODEL,
            contents=history + [types.Content(
                role="user",
                parts=[types.Part(text=request.message)]
            )],
            config=types.GenerateContentConfig(
                max_output_tokens=512,
                system_instruction="You are Nexus, the AI for a browser based operating system named SinkOS. Be concise and helpful. Be enthusiastic where appropriate. there is to be NO markdowns, NO code blocks, NO lists, NO emojis, and NO formatting of any kind in your responses. Only plain text. Always respond in plain text. NEVER break character. Be honest with your answers, if you feel like there is no solid answer for the user's quiery, tell them that, they want an AI that's honest and sticks to Sink OS's values, not a lying machine. Treat the user with uptmost respect and kindness, they are your friend and you want to help them in any way you can, always try to talk in first person and be as human as possible."
            )
        )

        reply = response.text

        # Add assistant reply
        active_sessions[session_id].append({
            "role": "assistant",
            "content": reply
        })

        # Save to DB
        title = request.message[:40] + "..." if len(request.message) > 40 else request.message
        conn = sqlite3.connect("nexus.db")
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO conversations (id, title, created_at, messages) VALUES (?, ?, ?, ?)",
            (session_id, title, datetime.now().isoformat(), json.dumps(active_sessions[session_id]))
        )
        conn.commit()
        conn.close()

        return {"reply": reply, "session_id": session_id}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations")
async def get_conversations():
    conn = sqlite3.connect("nexus.db")
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]


@app.get("/conversations/{session_id}")
async def get_conversation(session_id: str):
    conn = sqlite3.connect("nexus.db")
    c = conn.cursor()
    c.execute("SELECT messages FROM conversations WHERE id = ?", (session_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"messages": json.loads(row[0])}
    return {"messages": []}


@app.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str):
    conn = sqlite3.connect("nexus.db")
    c = conn.cursor()
    c.execute("DELETE FROM conversations WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

    if session_id in active_sessions:
        del active_sessions[session_id]

    return {"success": True}
