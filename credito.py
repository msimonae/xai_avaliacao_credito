# main.py
import os
import re
import io
import requests
from urllib.parse import urlparse, quote_plus, parse_qs
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from PIL import Image

import google.generativeai as genai

# Carrega as variáveis de ambiente
load_dotenv()

# Configuração da API do Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Inicializa FastAPI
app = FastAPI(
    title="API de Perguntas Inteligentes",
    description="Uma API que responde perguntas e gera resumos usando LLM (Gemini).",
    version="1.0.0"
)

# -------- MODELOS -------- #
class PerguntaRequest(BaseModel):
    pergunta: str

class ResumoRequest(BaseModel):
    texto: str
    max_tokens: Optional[int] = 200

class ImagemRequest(BaseModel):
    url: HttpUrl
    prompt: Optional[str] = "Descreva esta imagem em detalhes."

# -------- ROTAS -------- #

@app.get("/")
def home():
    return {"mensagem": "API rodando! Use /docs para ver a documentação interativa."}


@app.post("/perguntar/")
def perguntar(req: PerguntaRequest):
    """
    Recebe uma pergunta em texto e retorna uma resposta usando Gemini.
    """
    try:
        model = genai.GenerativeModel("gemini-pro")
        resposta = model.generate_content(req.pergunta)
        return {"pergunta": req.pergunta, "resposta": resposta.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resumir/")
def resumir(req: ResumoRequest):
    """
    Recebe um texto e retorna um resumo.
    """
    try:
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"Resuma o seguinte texto em até {req.max_tokens} palavras:\n\n{req.texto}"
        resposta = model.generate_content(prompt)
        return {"resumo": resposta.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analisar-imagem/")
def analisar_imagem(req: ImagemRequest):
    """
    Recebe uma URL de imagem e retorna uma descrição.
    """
    try:
        # Baixa a imagem
        response = requests.get(req.url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Não foi possível baixar a imagem.")

        image = Image.open(io.BytesIO(response.content))

        # Modelo multimodal
        model = genai.GenerativeModel("gemini-pro-vision")
        resposta = model.generate_content([req.prompt, image])

        return {"prompt": req.prompt, "descricao": resposta.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- EXECUÇÃO -------- #
# Rodar com: uvicorn main:app --reload
