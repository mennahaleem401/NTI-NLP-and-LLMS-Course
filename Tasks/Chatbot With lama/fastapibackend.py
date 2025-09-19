from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os

# --- FastAPI app ---
app = FastAPI()

# --- Groq client ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")
client = Groq(api_key=GROQ_API_KEY)


# --- Schema ---
class ChatRequest(BaseModel):
    message: str

# --- Endpoint ---
@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   
        messages=[
            {"role": "system", "content": "You are a helpful chatbot."},
            {"role": "user", "content": user_message}
        ],
        max_tokens=200,
    )

    bot_reply = completion.choices[0].message.content
    return {"reply": bot_reply}
