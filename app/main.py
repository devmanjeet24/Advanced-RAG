from fastapi import FastAPI
from app.routes import auth, rag, evaluation, ab_test, document, user, chat  # ← chat ADD karo yahan

app = FastAPI(title="Advanced RAG API", version="2.0.0")

app.include_router(auth.router,       prefix="/auth")
app.include_router(rag.router,        prefix="/rag")
app.include_router(evaluation.router, prefix="/eval")
app.include_router(ab_test.router,    prefix="/ab")
app.include_router(document.router,   prefix="/document")
app.include_router(user.router,       prefix="/user")
app.include_router(chat.router,       prefix="/chat")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "version": "2.0.0",
        "endpoints": {
            "auth":     ["/auth/register", "/auth/login"],
            "rag":      ["/rag/query", "/rag/stream", "/rag/transform"],
            "document": ["/document/upload"],
            "eval":     ["/eval/score", "/eval/pipeline"],
            "ab_test":  ["/ab/"],
            "user":     ["/user/me"],
            "chat":     ["/chat/", "/chat/stream", "/chat/quick"],
        }
    }