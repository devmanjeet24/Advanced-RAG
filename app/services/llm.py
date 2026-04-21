from groq import Groq
from app.core.config import settings

# Initialize Groq client once
client = Groq(api_key=settings.GROQ_API_KEY)

# MODEL = "llama3-70b-8192"
MODEL = "llama-3.3-70b-versatile"


def generate_answer(prompt: str, system: str = None, max_tokens: int = 512) -> str:
    messages = []

    if system:
        messages.append({"role": "system", "content": system})
    else:
        messages.append({
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer questions clearly and concisely "
                "based only on the provided context. If the context doesn't contain "
                "enough information, say so honestly."
            )
        })

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def generate_answer_stream(prompt: str, system: str = None):
    """Generator that yields chunks for streaming responses."""
    messages = []

    if system:
        messages.append({"role": "system", "content": system})
    else:
        messages.append({
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer questions clearly and concisely "
                "based only on the provided context."
            )
        })

    messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta