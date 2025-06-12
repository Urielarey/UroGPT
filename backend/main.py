from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uuid
from typing import Dict
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks

# Para hacer búsquedas web
from fastapi import Depends

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


# IMPORTANTE: Aquí vamos a simular la función web() que usarías con el plugin oficial OpenAI, 
# pero acá solo te pongo un stub que podés adaptar después al plugin real o alguna API de búsqueda.

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar para producción
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Historial de chats por sesión (simple en memoria)
session_histories: Dict[str, torch.Tensor] = {}

class Message(BaseModel):
    text: str
    session_id: str = None  # opcional para mantener sesiones

def es_pregunta_internet(texto: str) -> bool:
    keywords = ["quién", "qué", "cuándo", "dónde", "cómo", "por qué", "información", "últimas noticias", "clima", "resultado", "buscar", "internet"]
    texto = texto.lower()
    return any(k in texto for k in keywords)

async def busqueda_web(query: str) -> str:
    # Stub: Acá podés integrar la función real de web()
    # Por ahora simulamos respuesta
    # Más adelante, si querés, te ayudo a hacer scraping o llamar API Google/Bing
    return f"Resultado de búsqueda simulado para: '{query}'. Próximamente integrado con búsqueda real."

@app.post("/chat")
async def chat(message: Message):
    session_id = message.session_id or str(uuid.uuid4())
    user_text = message.text.strip()

    # Detectar si pregunta necesita búsqueda web
    if es_pregunta_internet(user_text):
        resultado_busqueda = await busqueda_web(user_text)
        return JSONResponse(content={"response": resultado_busqueda, "session_id": session_id})

    # Si no, seguimos con DialoGPT
    new_user_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors='pt')

    chat_history_ids = session_histories.get(session_id, None)

    if chat_history_ids is None:
        bot_input_ids = new_user_input_ids
    else:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=bot_input_ids.shape[-1] + 100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.6,
        eos_token_id=tokenizer.eos_token_id
    )

    session_histories[session_id] = chat_history_ids

    response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()

    if not response:
        response = "Lo siento, no entendí bien. ¿Podrías reformular?"

    return {"response": response, "session_id": session_id}

@app.post("/reset")
def reset_session(session: dict):
    session_id = session.get("session_id")
    if session_id and session_id in session_histories:
        del session_histories[session_id]
        return {"message": f"Sesión {session_id} reiniciada correctamente."}
    return {"message": "Sesión no encontrada o no proveída."}
