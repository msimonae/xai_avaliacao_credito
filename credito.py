import streamlit as st
import os, numpy as np, pandas as pd, joblib, shap, matplotlib.pyplot as plt
from anchor import anchor_tabular
import lime.lime_tabular
import eli5
from eli5 import format_as_html
from openai import OpenAI

st.set_page_config(page_title="Crédito com XAI", layout="wide")

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

# ------------------- OpenAI Key -------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    st.warning("⚠️ OPENAI_API_KEY não configurada. O feedback do LLM será pulado.")

# ------------------- UI -------------------
st.title("Previsão de Crédito e Explicabilidade (SHAP • LIME • ELI5 • Anchor)")

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
    VL_IMOVEIS = st.number_input('Valor dos Imóveis (R$)', min_value=0.0, value=300000.0, step=1000.0)
    OUTRA_RENDA = st.radio('Outra renda?', ['Sim', 'Não'], index=1)
    OUTRA_RENDA_VALOR = st.number_input('Valor Outra Renda (R$)', min_value=0.0, value=3000.0, step=100.0) if OUTRA_RENDA == 'Sim' else 0.0
    TEMPO_ULTIMO_EMPREGO_MESES = st.slider('Tempo Últ. Emprego (meses)', 0, 240, 18)
with col3:
    TRABALHANDO_ATUALMENTE = st.checkbox('Trabalhando atualmente?', value=True)
    ULTIMO_SALARIO = st.number_input('Último Salário (R$)', min_value=0.0, value=20400.0, step=100.0)
    QT_CARROS_input = st.multiselect('Qtd. Carros', [0,1,2,3,4,5], default=[1])
    VALOR_TABELA_CARROS = st.slider('Valor Tabela Carros (R$)', 0, 200000, 60000, step=5000)
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

    # ✅ Correção: reconstruir X_input_df e X_input_scaled
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

    exp_rec = ""  # acumulador para explicações

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

        razoes_shap = [
            f"{feature_names[j]} (contribuição SHAP: {contribs[j]:.2f}, valor: {X_input_df.iloc[0, j]})"
            for j in idx
        ]
        for r in razoes_shap:
            st.markdown(f"- {r}")

        exp_rec += f"Principais fatores (SHAP): {razoes_shap}\n"

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
        lime_features = [f for f, _ in lime_exp.as_list()]
        st.write(f"**LIME – Principais fatores:** {lime_features}")
        exp_rec += f"Principais fatores (LIME): {lime_features}\n"

        st.markdown("**Detalhe LIME:**")
        st.components.v1.html(lime_exp.as_html(), height=420, scrolling=True)
    except Exception as e:
        st.warning(f"Não foi possível gerar LIME: {e}")

    # ------------------- ELI5 -------------------
    try:
        eli5_expl = eli5.explain_prediction(lr_model, X_input_df.iloc[0], feature_names=feature_names)
        eli5_neg = [w.feature for w in eli5_expl.targets[0].feature_weights.neg]
        eli5_pos = [w.feature for w in eli5_expl.targets[0].feature_weights.pos]
        st.write(f"**ELI5 – Negativos:** {eli5_neg}")
        st.write(f"**ELI5 – Positivos:** {eli5_pos}")
        exp_rec += f"ELI5 negativos: {eli5_neg}, positivos: {eli5_pos}\n"

        st.markdown("**Detalhe ELI5:**")
        html_eli5 = format_as_html(eli5_expl)
        st.components.v1.html(html_eli5, height=420, scrolling=True)
    except Exception as e:
        st.warning(f"Não foi possível gerar ELI5: {e}")

    # ------------------- Anchor -------------------
    try:
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
        rule = anchor_exp.names()
        st.write(f"**Anchor – Regra mínima:** {rule}")
        st.write(f"Precisão: {anchor_exp.precision():.2f} | Cobertura: {anchor_exp.coverage():.2f}")
        exp_rec += f"Anchor: {rule}\n"
    except Exception as e:
        st.warning(f"Não foi possível gerar Anchor: {e}")

    # ------------------- Feedback do LLM -------------------
    if OPENAI_API_KEY:
        prompt = f"""
Você é um especialista em Machine Learning e XAI.
Com base nas explicações de SHAP, LIME, ELI5 e Anchor abaixo, escreva um feedback claro e amigável ao cliente
sobre os motivos do resultado e recomendações para aumentar as chances de aprovação futura.

Explicações:
{exp_rec}
"""
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)

            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um especialista em explicabilidade de crédito."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500
            )
            st.markdown("### 🔍 Feedback do Especialista")
            st.write(resp.choices[0].message['content'])
        except Exception as e:
            st.error(f"Erro ao gerar feedback com OpenAI: {e}")
    else:
        st.info("Configure OPENAI_API_KEY (st.secrets ou variável de ambiente) para habilitar o feedback do LLM.")
