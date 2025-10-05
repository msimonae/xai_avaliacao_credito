# credito.py
import os
import io
import requests
from dotenv import load_dotenv
from PIL import Image
import streamlit as st
import google.generativeai as genai

# Carrega variáveis de ambiente
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ----------- CONFIG STREAMLIT ----------- #
st.set_page_config(page_title="Explicabilidade de Crédito", layout="wide")
st.title("💳 Explicabilidade de Crédito com XAI + Gemini")

menu = st.sidebar.radio("Escolha a função:", ["Perguntar", "Resumir Texto", "Analisar Imagem", "Explicabilidade LIME"])

# ----------- GEMINI PERGUNTAS ----------- #
if menu == "Perguntar":
    pergunta = st.text_input("Digite sua pergunta:")
    if st.button("Enviar"):
        if pergunta.strip():
            model = genai.GenerativeModel("gemini-pro")
            resposta = model.generate_content(pergunta)
            st.success(resposta.text)

# ----------- RESUMO DE TEXTO ----------- #
elif menu == "Resumir Texto":
    texto = st.text_area("Digite o texto para resumir:")
    max_tokens = st.slider("Máx. de palavras no resumo", 50, 500, 200)
    if st.button("Resumir"):
        if texto.strip():
            model = genai.GenerativeModel("gemini-pro")
            prompt = f"Resuma o seguinte texto em até {max_tokens} palavras:\n\n{texto}"
            resposta = model.generate_content(prompt)
            st.info(resposta.text)

# ----------- ANÁLISE DE IMAGEM ----------- #
elif menu == "Analisar Imagem":
    url = st.text_input("Insira a URL da imagem:")
    prompt = st.text_input("Prompt para análise:", "Descreva esta imagem em detalhes.")
    if st.button("Analisar Imagem"):
        if url.strip():
            try:
                response = requests.get(url)
                image = Image.open(io.BytesIO(response.content))
                st.image(image, caption="Imagem carregada", use_container_width=True)

                model = genai.GenerativeModel("gemini-pro-vision")
                resposta = model.generate_content([prompt, image])
                st.success(resposta.text)
            except Exception as e:
                st.error(f"Erro ao processar imagem: {e}")

# ----------- EXPLICABILIDADE LIME ----------- #
elif menu == "Explicabilidade LIME":
    st.subheader("📊 Explicabilidade com LIME")

    # 🔹 Exemplo de saída bruta do LIME (simulação — aqui você deve integrar o resultado real do LIME)
    lime_rules = [
        "feature_property_value <= 185000, valor observado: 100000 → impacto negativo",
        "feature_car_value <= 50000, valor observado: 45000 → impacto negativo",
        "feature_last_job_time <= 14, valor observado: 5 → impacto negativo",
        "feature_car_count <= 1, valor observado: 1 → impacto negativo",
        "feature_salary <= 6100, valor observado: 5400 → impacto negativo"
    ]

    # Mostrando saída original do LIME
    st.write("### 🔎 Saída Original (Bruta do LIME)")
    for regra in lime_rules:
        st.text(regra)

    # Feedback do Especialista (LLM interpretando as regras do LIME)
    st.write("### 🧑‍🏫 Feedback do Especialista (LLM)")
    prompt = f"""
    Explique de forma acessível os seguintes resultados do LIME aplicados à análise de crédito.
    Mantenha um tom explicativo, destacando riscos e oportunidades:

    {lime_rules}
    """
    model = genai.GenerativeModel("gemini-pro")
    resposta = model.generate_content(prompt)
    st.info(resposta.text)
