import os
import uuid
import mimetypes
import traceback
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Header, File, UploadFile, Form
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


# ─── File / image / PDF attachments ────────────────────────────────────────
ALLOWED_ATTACHMENT_TYPES = {
    "image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp",
    "application/pdf",
}
MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024        # 8MB per file
MAX_ATTACHMENTS_PER_MESSAGE = 4
ATTACHMENTS_BUCKET = "nexus-attachments"       # must exist in Supabase Storage — see notes


@app.post("/chat/upload")
async def chat_upload(
    message: str = Form(""),
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    authorization: Optional[str] = Header(None),
):
    user_id = get_user_from_token(authorization)
    try:
        if len(files) > MAX_ATTACHMENTS_PER_MESSAGE:
            raise HTTPException(
                status_code=400,
                detail=f"Max {MAX_ATTACHMENTS_PER_MESSAGE} files per message",
            )

        session_id = session_id or str(uuid.uuid4())

        if session_id not in active_sessions:
            row = supabase.table("nexus_conversations") \
                .select("messages") \
                .eq("id", session_id) \
                .eq("user_id", user_id) \
                .execute()
            active_sessions[session_id] = row.data[0]["messages"] if row.data else []

        gemini_parts = []
        stored_attachments = []

        for f in files:
            # Strip any ";charset=..." etc — same fix as the STT mime-type bug.
            content_type = (f.content_type or "").split(";")[0].strip().lower()
            if content_type not in ALLOWED_ATTACHMENT_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {content_type or 'unknown'} ({f.filename})",
                )

            data = await f.read()
            if not data:
                continue
            if len(data) > MAX_ATTACHMENT_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"{f.filename} exceeds the 8MB limit",
                )

            # Persist to Supabase Storage so it's still viewable when the
            # conversation is reloaded later (Gemini only sees it this turn).
            ext = mimetypes.guess_extension(content_type) or ""
            storage_path = f"{user_id}/{uuid.uuid4()}{ext}"
            try:
                supabase.storage.from_(ATTACHMENTS_BUCKET).upload(
                    storage_path,
                    data,
                    {"content-type": content_type},
                )
            except Exception:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Failed to store {f.filename}. Make sure the "
                        f"'{ATTACHMENTS_BUCKET}' bucket exists in Supabase Storage."
                    ),
                )

            public_url = supabase.storage.from_(ATTACHMENTS_BUCKET).get_public_url(storage_path)
            # Some supabase-py versions return {"publicUrl": "..."} instead of a bare string.
            if isinstance(public_url, dict):
                public_url = public_url.get("publicUrl") or public_url.get("public_url")

            stored_attachments.append({
                "url": public_url,
                "mime_type": content_type,
                "name": f.filename,
            })

            gemini_parts.append(types.Part.from_bytes(data=data, mime_type=content_type))

        if message.strip():
            gemini_parts.append(types.Part(text=message.strip()))
        elif not gemini_parts:
            raise HTTPException(status_code=400, detail="No message or files provided")

        # History is replayed as text only — attachments from earlier turns
        # are not re-uploaded to Gemini on later turns.
        history = []
        for msg in active_sessions[session_id]:
            role = "model" if msg["role"] == "assistant" else "user"
            history.append(types.Content(role=role, parts=[types.Part(text=msg.get("content") or "")]))

        response = client.models.generate_content(
            model=MODEL,
            contents=history + [types.Content(role="user", parts=gemini_parts)],
            config=types.GenerateContentConfig(
                max_output_tokens=768,
                system_instruction="You are Nexus, the AI for a browser based operating system named SinkOS. Be concise and helpful. Be enthusiastic where appropriate. there is to be NO markdowns, NO code blocks, NO lists, NO emojis, and NO formatting of any kind in your responses. Only plain text. Always respond in plain text. NEVER break character. Be honest with your answers, if you feel like there is no solid answer for the user's quiery, tell them that, they want an AI that's honest and sticks to Sink OS's values, not a lying machine. Treat the user with uptmost respect and kindness, they are your friend and you want to help them in any way you can, always try to talk in first person and be as human as possible."
            )
        )

        reply = response.text
        user_content = message.strip()

        active_sessions[session_id].append({
            "role": "user",
            "content": user_content,
            "attachments": stored_attachments,
        })
        active_sessions[session_id].append({"role": "assistant", "content": reply})

        title_source = user_content or (stored_attachments[0]["name"] if stored_attachments else "Attachment")
        title = title_source[:40] + "..." if len(title_source) > 40 else title_source
        supabase.table("nexus_conversations").upsert({
            "id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "messages": active_sessions[session_id]
        }).execute()

        return {"reply": reply, "session_id": session_id, "attachments": stored_attachments}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
# Receives a recorded utterance (webm/opus from MediaRecorder) and transcribes
# it via Gemini 2.5 Flash Lite. Auth-gated the same way as /chat. No audio is
# ever persisted server-side — it's transcribed in-memory and discarded.
@app.post("/voice/stt")
async def voice_stt(audio: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)  # noqa: F841 (kept for auth gating + future per-user logging)
    try:
        audio_bytes = await audio.read()
        if not audio_bytes or len(audio_bytes) < 800:
            # Too small to be a real utterance — don't waste a Gemini call on
            # noise/silence, and don't risk it hallucinating a transcript.
            return {"transcript": ""}

        # IMPORTANT: strip codec parameters (e.g. "audio/webm;codecs=opus" -> "audio/webm").
        # Gemini's inline_data mime_type expects a bare MIME type; passing the
        # codecs suffix through degrades or breaks audio parsing, which was
        # causing confidently-wrong ("hallucinated") transcripts.
        raw_mime = audio.content_type or "audio/webm"
        mime_type = raw_mime.split(";")[0].strip()

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                        types.Part(text=(
                            "Transcribe ONLY the words actually spoken in this audio clip, exactly as spoken. "
                            "Reply with ONLY the transcript text — no preamble, no quotes, no commentary, "
                            "no translation, no correction of grammar. "
                            "Do not guess, invent, or complete words you cannot clearly hear. "
                            "If the audio is silent, contains no intelligible speech, or is too short/unclear "
                            "to transcribe with confidence, reply with an empty string and nothing else."
                        )),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=256,
                temperature=0.0,  # deterministic, minimizes creative "filling in" on weak audio
            ),
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
