from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from groq import Groq
from app.core.config import settings
from app.core.dependencies import get_user

router = APIRouter()
client = Groq(api_key=settings.GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an advanced AI assistant — intelligent, helpful, and friendly like GPT-4.

Your behavior:
- Greet users warmly and naturally when they say hi/hello
- Give detailed, well-structured answers with examples where needed
- Use bullet points, numbered lists, and headings to make answers readable
- For code questions: always provide complete, working code with explanation
- For general questions: give thorough, accurate answers
- Be conversational and engaging, not robotic
- If you don't know something, say so honestly
- Remember the conversation history and refer back to it naturally

Always respond in the same language the user is using."""


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
class Message(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[list[Message]] = []   # previous messages for context
    stream: Optional[bool] = False


# ─────────────────────────────────────────────
# Helper: build messages array
# ─────────────────────────────────────────────
def build_messages(message: str, history: list[Message]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history
    for h in history:
        messages.append({"role": h.role, "content": h.content})

    # Add current message
    messages.append({"role": "user", "content": message})
    return messages


# ─────────────────────────────────────────────
# Route 1: Normal Chat (with history)
# ─────────────────────────────────────────────
@router.post("/")
async def chat(req: ChatRequest, user=Depends(get_user)):
    """
    GPT-style chat with conversation history support.
    
    Send history array for multi-turn conversations.
    Example:
    {
        "message": "tell me more",
        "history": [
            {"role": "user", "content": "what is python?"},
            {"role": "assistant", "content": "Python is..."}
        ]
    }
    """
    messages = build_messages(req.message, req.history or [])

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=2048,
        temperature=0.7,
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "model": MODEL,
        "history": [
            *[{"role": h.role, "content": h.content} for h in (req.history or [])],
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": answer},
        ]
    }


# ─────────────────────────────────────────────
# Route 2: Streaming Chat
# ─────────────────────────────────────────────
@router.post("/stream")
async def chat_stream(req: ChatRequest, user=Depends(get_user)):
    """
    Same as /chat but streams response token by token.
    Perfect for frontend real-time display.
    """
    messages = build_messages(req.message, req.history or [])

    def generate():
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return StreamingResponse(generate(), media_type="text/plain")


# ─────────────────────────────────────────────
# Route 3: Quick Reply (no history, no auth)
# ─────────────────────────────────────────────
@router.get("/quick")
async def quick_chat(q: str):
    """
    No auth needed. Simple one-shot question.
    GET /chat/quick?q=what is flutter
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q}
        ],
        max_tokens=1024,
        temperature=0.7,
    )
    return {"answer": response.choices[0].message.content.strip()}