from fastapi import FastAPI, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uuid
from typing import Dict

# Crear la app primero
app = FastAPI()

# Montar archivos estáticos (frontend)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar esto en producción
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Historial de sesiones (simple en memoria)
session_histories: Dict[str, torch.Tensor] = {}

# Modelo para mensaje
class Message(BaseModel):
    text: str
    session_id: str = None

# Detectar si requiere búsqueda web (simulada)
def es_pregunta_internet(texto: str) -> bool:
    keywords = ["quién", "qué", "cuándo", "dónde", "cómo", "por qué", "información", "últimas noticias", "clima", "resultado", "buscar", "internet"]
    texto = texto.lower()
    return any(k in texto for k in keywords)

async def busqueda_web(query: str) -> str:
    return f"Resultado simulado para: '{query}'. (Integración real pendiente)"

# Ruta para chatear
@app.post("/chat")
async def chat(message: Message):
    session_id = message.session_id or str(uuid.uuid4())
    user_text = message.text.strip()

    if es_pregunta_internet(user_text):
        resultado_busqueda = await busqueda_web(user_text)
        return JSONResponse(content={"response": resultado_busqueda, "session_id": session_id})

    new_user_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = session_histories.get(session_id, None)

    bot_input_ids = new_user_input_ids if chat_history_ids is None else torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

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
        response = "Lo siento, no entendí bien. ¿Podés reformular?"

    return {"response": response, "session_id": session_id}

# Ruta para reiniciar sesión
@app.post("/reset")
def reset_session(session: dict):
    session_id = session.get("session_id")
    if session_id and session_id in session_histories:
        del session_histories[session_id]
        return {"message": f"Sesión {session_id} reiniciada correctamente."}
    return {"message": "Sesión no encontrada o no proveída."}
