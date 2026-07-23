import os
import uuid
import traceback
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google import genai
from google.genai import types
from supabase import create_client, Client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash-lite"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def get_user_from_token(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        user_resp = supabase.auth.get_user(token)
        if not user_resp or not user_resp.user:
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        return user_resp.user.id
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

active_sessions: dict = {}

class MessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.post("/chat")
async def chat(request: MessageRequest, authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    try:
        session_id = request.session_id or str(uuid.uuid4())

        if session_id not in active_sessions:
            row = supabase.table("nexus_conversations") \
                .select("messages") \
                .eq("id", session_id) \
                .eq("user_id", user_id) \
                .execute()
            active_sessions[session_id] = row.data[0]["messages"] if row.data else []

        active_sessions[session_id].append({"role": "user", "content": request.message})

        history = []
        messages = active_sessions[session_id]
        for msg in messages[:-1]:
            role = "model" if msg["role"] == "assistant" else "user"
            history.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        response = client.models.generate_content(
            model=MODEL,
            contents=history + [types.Content(role="user", parts=[types.Part(text=request.message)])],
            config=types.GenerateContentConfig(
                max_output_tokens=512,
                system_instruction="You are Nexus, the AI for a browser based operating system named SinkOS. Be concise and helpful. Be enthusiastic where appropriate. there is to be NO markdowns, NO code blocks, NO lists, NO emojis, and NO formatting of any kind in your responses. Only plain text. Always respond in plain text. NEVER break character. Be honest with your answers, if you feel like there is no solid answer for the user's quiery, tell them that, they want an AI that's honest and sticks to Sink OS's values, not a lying machine. Treat the user with uptmost respect and kindness, they are your friend and you want to help them in any way you can, always try to talk in first person and be as human as possible."
            )
        )

        reply = response.text
        active_sessions[session_id].append({"role": "assistant", "content": reply})

        title = request.message[:40] + "..." if len(request.message) > 40 else request.message
        supabase.table("nexus_conversations").upsert({
            "id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "messages": active_sessions[session_id]
        }).execute()

        return {"reply": reply, "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Voice Mode: Speech-to-Text ────────────────────────────────────────────
# Receives a recorded utterance (webm/opus from MediaRecorder) and transcribes
# it via Gemini 2.5 Flash Lite. Auth-gated the same way as /chat. No audio is
# ever persisted server-side — it's transcribed in-memory and discarded.
@app.post("/voice/stt")
async def voice_stt(audio: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)  # noqa: F841 (kept for auth gating + future per-user logging)
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            return {"transcript": ""}

        mime_type = audio.content_type or "audio/webm"

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                        types.Part(text=(
                            "Transcribe the spoken audio exactly as spoken. "
                            "Reply with ONLY the transcript text — no preamble, no quotes, "
                            "no commentary. If no speech is detected, reply with an empty string."
                        )),
                    ],
                )
            ],
            config=types.GenerateContentConfig(max_output_tokens=256),
        )

        transcript = (response.text or "").strip()
        return {"transcript": transcript}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations")
async def get_conversations(authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    row = supabase.table("nexus_conversations") \
        .select("id, title, created_at") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .execute()
    return row.data


@app.get("/conversations/{session_id}")
async def get_conversation(session_id: str, authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    row = supabase.table("nexus_conversations") \
        .select("messages") \
        .eq("id", session_id) \
        .eq("user_id", user_id) \
        .execute()
    if row.data:
        return {"messages": row.data[0]["messages"]}
    return {"messages": []}


@app.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str, authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    supabase.table("nexus_conversations") \
        .delete() \
        .eq("id", session_id) \
        .eq("user_id", user_id) \
        .execute()

    if session_id in active_sessions:
        del active_sessions[session_id]

    return {"success": True}
