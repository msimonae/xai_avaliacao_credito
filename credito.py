import streamlit as st
import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from anchor import anchor_tabular
import lime.lime_tabular
import eli5
from eli5 import format_as_html
from openai import OpenAI
import re

# Função para formatar valores monetários
def format_currency(value):
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Função para processar e humanizar as regras do LIME
def process_lime_rule(rule):
    # Dicionário para traduzir nomes de features
    feature_translations = {
        'VL_IMOVEIS': 'o valor dos seus imóveis',
        'VALOR_TABELA_CARROS': 'o valor de tabela dos seus carros',
        'TEMPO_ULTIMO_EMPREGO_MESES': 'o seu tempo de último emprego',
        'ULTIMO_SALARIO': 'o seu último salário',
        'QT_CARROS': 'a quantidade de carros',
        'OUTRA_RENDA_VALOR': 'o valor da sua outra renda',
        'UF': 'a sua UF',
    }

    # Extrai o nome da feature
    feature_name = rule.split()[0].replace('<', '').replace('>', '').replace('<=', '').replace('>=', '')
    translated_name = feature_translations.get(feature_name, feature_name)
    
    # Humaniza as faixas de valores
    if '<' in rule and '<=' in rule:
        # Padrão: 0.00 < VL_IMOVEIS <= 185000.00
        values = re.findall(r'(\d+\.?\d*)', rule)
        if len(values) >= 2:
            lower_bound = float(values[0])
            upper_bound = float(values[1])
            if feature_name in ['VL_IMOVEIS', 'VALOR_TABELA_CARROS', 'ULTIMO_SALARIO', 'OUTRA_RENDA_VALOR']:
                return f"O valor de {translated_name} está entre {format_currency(lower_bound)} e {format_currency(upper_bound)}."
            else:
                return f"O valor de {translated_name} está entre {lower_bound} e {upper_bound}."
    
    # Humaniza regras de "menor ou igual"
    elif '<=' in rule:
        # Padrão: TEMPO_ULTIMO_EMPREGO_MESES <= 14.00
        value = float(re.search(r'<= (\d+\.?\d*)', rule).group(1))
        if feature_name in ['VL_IMOVEIS', 'ULTIMO_SALARIO', 'VALOR_TABELA_CARROS', 'OUTRA_RENDA_VALOR']:
            return f"O valor de {translated_name} é igual ou menor que {format_currency(value)}."
        else:
            return f"O valor de {translated_name} é igual ou menor que {value}."
    
    # Humaniza regras de "maior"
    elif '>' in rule:
        # Padrão: TEMPO_ULTIMO_EMPREGO_MESES > 14.00
        value = float(re.search(r'> (\d+\.?\d*)', rule).group(1))
        if feature_name in ['VL_IMOVEIS', 'ULTIMO_SALARIO', 'VALOR_TABELA_CARROS', 'OUTRA_RENDA_VALOR']:
            return f"O valor de {translated_name} é maior que {format_currency(value)}."
        else:
            return f"O valor de {translated_name} é maior que {value}."

    return rule  # Retorna a regra original se não houver um padrão correspondente

st.set_page_config(page_title="Crédito com XAI", layout="wide")

# --- Centralizar a configuração da API Key e a criação do cliente ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    st.warning("⚠️ OPENAI_API_KEY não configurada. O feedback do LLM não estará disponível.")

# ------------------- Carregar modelos/dados -------------------
feature_names = [
    'UF', 'ESCOLARIDADE', 'ESTADO_CIVIL', 'QT_FILHOS', 'CASA_PROPRIA',
    'QT_IMOVEIS', 'VL_IMOVEIS', 'OUTRA_RENDA', 'OUTRA_RENDA_VALOR',
    'TEMPO_ULTIMO_EMPREGO_MESES', 'TRABALHANDO_ATUALMENTE', 'ULTIMO_SALARIO',
    'QT_CARROS', 'VALOR_TABELA_CARROS', 'FAIXA_ETARIA'
]

try:
    scaler = joblib.load('scaler.pkl')
    lr_model = joblib.load('modelo_regressao.pkl')
    X_train_scaled = joblib.load('X_train_scaled.pkl')
    X_train_raw = joblib.load('X_train.pkl')
    if isinstance(X_train_raw, np.ndarray):
        X_train_df = pd.DataFrame(X_train_raw, columns=feature_names)
    else:
        if list(X_train_raw.columns) != feature_names:
            X_train_df = X_train_raw[feature_names]
        else:
            X_train_df = X_train_raw
except Exception as e:
    st.error(f"Erro ao carregar modelos/dados: {e}")
    st.stop()

# ------------------- UI -------------------
st.title("Previsão de Crédito e Explicabilidade (XAI)")

ufs = ['SP', 'MG', 'SC', 'PR', 'RJ']
escolaridades = ['Superior Cursando', 'Superior Completo', 'Segundo Grau Completo']
estados_civis = ['Solteiro', 'Casado', 'Divorciado']
faixas_etarias = ['18-25', '26-35', '36-45', '46-60', 'Acima de 60']

col1, col2, col3 = st.columns(3)
with col1:
    UF = st.selectbox('UF', ufs, index=0)
    ESCOLARIDADE = st.selectbox('Escolaridade', escolaridades, index=1)
    ESTADO_CIVIL = st.selectbox('Estado Civil', estados_civis, index=0)
    QT_FILHOS = st.number_input('Qtd. Filhos', min_value=0, value=1)
    CASA_PROPRIA = st.radio('Casa Própria?', ['Sim', 'Não'], index=0)
with col2:
    QT_IMOVEIS = st.number_input('Qtd. Imóveis', min_value=0, value=1)

    VL_IMOVEIS_str = st.text_input('Valor dos Imóveis (R$)', value="100000")
    try:
        VL_IMOVEIS = float(VL_IMOVEIS_str.replace("R$", "").replace(".", "").replace(",", "."))
    except:
        VL_IMOVEIS = 0.0

    OUTRA_RENDA = st.radio('Outra renda?', ['Sim', 'Não'], index=1)

    OUTRA_RENDA_VALOR = 0.0
    if OUTRA_RENDA == 'Sim':
        OUTRA_RENDA_VALOR_str = st.text_input('Valor Outra Renda (R$)', value="2000")
        try:
            OUTRA_RENDA_VALOR = float(OUTRA_RENDA_VALOR_str.replace("R$", "").replace(".", "").replace(",", "."))
        except:
            OUTRA_RENDA_VALOR = 0.0

    TEMPO_ULTIMO_EMPREGO_MESES = st.slider('Tempo Últ. Emprego (meses)', 0, 240, 5)

with col3:
    TRABALHANDO_ATUALMENTE = st.radio('Trabalhando atualmente?', ['Sim', 'Não'], index=0)
    trabalhando_flag = 1 if TRABALHANDO_ATUALMENTE == 'Sim' else 0

    if TRABALHANDO_ATUALMENTE == 'Sim':
        ULTIMO_SALARIO_str = st.text_input('Último Salário (R$)', value="5400")
        try:
            ULTIMO_SALARIO = float(
                ULTIMO_SALARIO_str.replace("R$", "").replace(".", "").replace(",", ".")
            )
        except:
            ULTIMO_SALARIO = 0.0
    else:
        ULTIMO_SALARIO = 0.0

    QT_CARROS_input = st.multiselect('Qtd. Carros', [0,1,2,3,4,5], default=[1])
    VALOR_TABELA_CARROS = st.slider('Valor Tabela Carros (R$)', 0, 200000, 45000, step=5000)
    FAIXA_ETARIA = st.radio('Faixa Etária', faixas_etarias, index=2)

if st.button("Verificar Crédito"):
    # ------------------- Montar dados do input -------------------
    uf_map = {label: i for i, label in enumerate(ufs)}
    escolaridade_map = {label: i for i, label in enumerate(escolaridades)}
    estado_civil_map = {label: i for i, label in enumerate(estados_civis)}
    faixa_etaria_map = {label: i for i, label in enumerate(faixas_etarias)}
    
    novos_dados = [
        uf_map[UF], escolaridade_map[ESCOLARIDADE], estado_civil_map[ESTADO_CIVIL], QT_FILHOS,
        1 if CASA_PROPRIA == 'Sim' else 0, QT_IMOVEIS, VL_IMOVEIS,
        1 if OUTRA_RENDA == 'Sim' else 0, OUTRA_RENDA_VALOR, TEMPO_ULTIMO_EMPREGO_MESES,
        1 if TRABALHANDO_ATUALMENTE else 0, ULTIMO_SALARIO, len(QT_CARROS_input),
        VALOR_TABELA_CARROS, faixa_etaria_map[FAIXA_ETARIA]
    ]

    X_input_df = pd.DataFrame([novos_dados], columns=feature_names)
    X_input_scaled = scaler.transform(X_input_df)
    X_input_scaled_df = pd.DataFrame(X_input_scaled, columns=feature_names)

    # Predição
    y_pred = lr_model.predict(X_input_scaled)[0]
    proba = getattr(lr_model, "predict_proba", lambda x: np.array([[1,0]]))(X_input_scaled)[0][1]
    resultado_texto = 'Aprovado' if y_pred == 1 else 'Recusado'
    cor = 'green' if y_pred == 1 else 'red'
    st.markdown(f"### Resultado: <span style='color:{cor}; font-weight:700'>{resultado_texto}</span>", unsafe_allow_html=True)
    st.write(f"Probabilidade de Aprovação: **{proba:.2%}**")

    # Acumuladores para explicações
    exp_rec_shap = ""
    exp_rec_lime = ""
    exp_rec_anchor = ""

    # ------------------- SHAP -------------------
    try:
        st.markdown("**Explicação com SHAP (Impacto das Features na Predição Atual):**")
        explainer = shap.TreeExplainer(lr_model)
        sv_scaled = explainer(X_input_scaled_df)

        sv_plot = shap.Explanation(
            values=sv_scaled.values[0],
            base_values=sv_scaled.base_values[0],
            data=X_input_df.iloc[0].values,
            feature_names=feature_names
        )

        fig_waterfall = plt.figure()
        shap.plots.waterfall(sv_plot, show=False, max_display=10)
        st.pyplot(fig_waterfall)
        plt.close(fig_waterfall)

        contribs = sv_scaled.values[0]
        if y_pred == 0:
            idx = np.argsort(contribs)[:3]
            st.write("**Principais fatores que influenciaram a recusa:**")
        else:
            idx = np.argsort(contribs)[-3:]
            st.write("**Principais fatores que influenciaram a aprovação:**")

        razoes_shap_list = []
        for j in idx:
            feature_name = feature_names[j]
            contrib = contribs[j]
            val = X_input_df.iloc[0, j]
            # --- CORREÇÃO: Humanizamos o SHAP aqui no Python, garantindo a formatação ---
            if feature_name in ['VL_IMOVEIS', 'ULTIMO_SALARIO', 'VALOR_TABELA_CARROS', 'OUTRA_RENDA_VALOR']:
                val_str = format_currency(val)
                razoes_shap_list.append(f"{feature_name}: contribuição de {contrib:.2f}, com um valor de {val_str}.")
            else:
                razoes_shap_list.append(f"{feature_name}: contribuição de {contrib:.2f}, com um valor de {val}.")
            # --- FIM DA CORREÇÃO ---
        
        # Envia a lista de frases humanizadas para o LLM
        exp_rec_shap = "\n".join(razoes_shap_list)

        for r in razoes_shap_list:
            st.markdown(f"- {r}")

    except Exception as e:
        st.warning(f"Não foi possível gerar SHAP: {e}")

    # ------------------- LIME -------------------
    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_df.values,
            feature_names=feature_names,
            class_names=['Recusado', 'Aprovado'],
            mode='classification'
        )
        lime_exp = lime_explainer.explain_instance(
            X_input_df.values[0],
            lambda x: lr_model.predict_proba(scaler.transform(pd.DataFrame(x, columns=feature_names))),
            num_features=5
        )
        lime_features = lime_exp.as_list()
        
        # --- CORREÇÃO: Humanizamos o LIME aqui no Python, garantindo a formatação ---
        processed_lime_features = []
        for rule, contrib in lime_features:
            human_rule = process_lime_rule(rule)
            processed_lime_features.append(f"{human_rule} Teve uma contribuição de {contrib:.2f}.")

        exp_rec_lime = "\n".join(processed_lime_features)
        
        st.write(f"**LIME – Principais fatores:** {lime_features}")
        
    except Exception as e:
        st.warning(f"Não foi possível gerar LIME: {e}")

    # ------------------- ELI5 -------------------
    try:
        eli5_expl = eli5.explain_prediction(lr_model, X_input_df.iloc[0], feature_names=feature_names)
        eli5_neg = [w.feature for w in eli5_expl.targets[0].feature_weights.neg]
        eli5_pos = [w.feature for w in eli5_expl.targets[0].feature_weights.pos]
        st.write(f"**ELI5 – Negativos:** {eli5_neg}")
        st.write(f"**ELI5 – Positivos:** {eli5_pos}")

        st.markdown("**Detalhe ELI5:**")
        html_eli5 = format_as_html(eli5_expl)
        st.components.v1.html(html_eli5, height=420, scrolling=True)
    except Exception as e:
        st.warning(f"Não foi possível gerar ELI5: {e}")

    # ------------------- Anchor -------------------
    try:
        st.markdown("**Explicação com Anchor (Regras Mínimas):**")
        def predict_fn_anchor(arr2d):
            df = pd.DataFrame(arr2d, columns=feature_names)
            scaled = scaler.transform(df)
            return lr_model.predict(scaled)

        anchor_explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=['Recusado', 'Aprovado'],
            feature_names=feature_names,
            train_data=X_train_df.values
        )
        anchor_exp = anchor_explainer.explain_instance(
            X_input_df.values[0], predict_fn_anchor, threshold=0.95
        )
        
        rule = " E ".join(anchor_exp.names())
        st.write(f"**Anchor – Regra que ancora a predição:** Se *{rule}*, então o resultado é **{resultado_texto}**.")
        st.write(f"Precisão da regra: {anchor_exp.precision():.2f} | Cobertura da regra: {anchor_exp.coverage():.2f}")
        exp_rec_anchor = f"Regra Anchor: {rule}"

    except Exception as e:
        st.warning(f"Não foi possível gerar a explicação Anchor: {e}")

    # ------------------- Feedback do LLM -------------------
    if client:
        # Prompt otimizado e mais claro
        prompt = f"""
Você é um Cientista de Dados Sênior, especialista em explicar os resultados de modelos de Machine Learning para clientes de forma clara, objetiva e humana.
O modelo de análise de crédito previu o resultado '{resultado_texto}' para um cliente.

Aqui estão as explicações técnicas sobre os fatores que mais influenciaram essa decisão:
- **SHAP (Contribuição dos atributos):**
{exp_rec_shap}
- **LIME (Regras de decisão):**
{exp_rec_lime}

Com base nas informações do **SHAP** e **LIME**, crie um feedback amigável para o cliente, seguindo as instruções abaixo:

1.  **Análise do Resultado:** De forma amigável e empática, explique os principais motivos que levaram à decisão. Mencione os fatores do SHAP. Para o LIME, use as regras fornecidas para explicar a decisão.

2.  **Pontos a Melhorar (se o resultado for 'Recusado')**: Se o crédito foi recusado, forneça 2 ou 3 dicas práticas sobre como o cliente pode melhorar seu perfil.

3.  **Estrutura:** Divida sua resposta em tópicos, como "Análise do seu Perfil Financeiro" e "Recomendações".

Seja direto, empático e construtivo. Evite qualquer tipo de concatenação de palavras. Não inclua informações sobre a explicação do Anchor no seu feedback.
"""
        try:
            with st.spinner("Gerando feedback personalizado..."):
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Você é um analista de crédito sênior e especialista em comunicação com clientes."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                st.markdown("### 🔍 Feedback do Especialista feito pelo LLM")
                feedback_content = resp.choices[0].message.content
                st.write(feedback_content)
        except Exception as e:
            st.error(f"Erro ao gerar feedback com a OpenAI: {e}")
