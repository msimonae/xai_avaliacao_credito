import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap # Importa a biblioteca SHAP
import matplotlib.pyplot as plt # Para plots do SHAP
from anchor import anchor_tabular # Importa a biblioteca Anchor

# Definir nomes das features na ordem correta.
# Esta ordem DEVE corresponder à ordem das colunas nos dados de treino originais
# e à ordem em que 'novos_dados_list' é construída.
feature_names = [
    'UF', 'ESCOLARIDADE', 'ESTADO_CIVIL', 'QT_FILHOS', 'CASA_PROPRIA',
    'QT_IMOVEIS', 'VL_IMOVEIS', 'OUTRA_RENDA', 'OUTRA_RENDA_VALOR',
    'TEMPO_ULTIMO_EMPREGO_MESES', 'TRABALHANDO_ATUALMENTE', 'ULTIMO_SALARIO',
    'QT_CARROS', 'VALOR_TABELA_CARROS', 'FAIXA_ETARIA'
]

# Carregar os modelos e dados de treino
try:
    scaler = joblib.load('/content/drive/MyDrive/Colab Notebooks/TCC/scaler.pkl')
    lr_model = joblib.load('/content/drive/MyDrive/Colab Notebooks/TCC/modelo_regressao.pkl') # Assumindo que é um modelo XGBoost ou similar

    # Carregar dados de treino (não escalados) para o Anchor e potencialmente SHAP (background)
    # Se X_train.pkl já é um DataFrame com as colunas corretas, ótimo.
    # Se for um array NumPy, precisa ter as colunas na mesma ordem de feature_names.
    X_train_raw = joblib.load('/content/drive/MyDrive/Colab Notebooks/TCC/X_train.pkl')
    if isinstance(X_train_raw, np.ndarray):
        X_train_df = pd.DataFrame(X_train_raw, columns=feature_names)
    elif isinstance(X_train_raw, pd.DataFrame):
        # Verificar se as colunas do DataFrame carregado correspondem a feature_names
        if list(X_train_raw.columns) == feature_names:
            X_train_df = X_train_raw
        else:
            # Tentar reordenar ou alertar sobre a discrepância
            st.warning("Colunas em X_train.pkl não correspondem à ordem esperada. Tentando reordenar.")
            try:
                X_train_df = X_train_raw[feature_names]
            except KeyError:
                st.error("Erro fatal: Colunas em X_train.pkl não encontradas ou na ordem incorreta. Verifique feature_names e o arquivo X_train.pkl.")
                st.stop()
    else:
        st.error("Formato de X_train.pkl não reconhecido.")
        st.stop()

    # Carregar dados de treino escalados (para SHAP LinearExplainer, se aplicável)
    # X_train_scaled_raw = joblib.load('/content/drive/MyDrive/Colab Notebooks/TCC/X_train_scaled.pkl')
    # if isinstance(X_train_scaled_raw, np.ndarray):
    #     X_train_scaled_df = pd.DataFrame(X_train_scaled_raw, columns=feature_names)
    # else:
    #     X_train_scaled_df = X_train_scaled_raw


except FileNotFoundError:
    st.error("Erro: Um ou mais arquivos de modelo/dados (.pkl) não foram encontrados. Verifique os caminhos.")
    st.stop()
except Exception as e:
    st.error(f"Erro ao carregar modelos ou dados de treino: {e}")
    st.stop()


st.title("Previsão de Crédito e Explicação")

# Opções para os campos categóricos (mantidas do original)
ufs = ['SP', 'MG', 'SC', 'PR', 'RJ']
escolaridades = ['Superior Cursando', 'Superior Completo', 'Segundo Grau Completo']
estados_civis = ['Solteiro', 'Casado', 'Divorciado']
faixas_etarias = ['18-25', '26-35', '36-45', '46-60', 'Acima de 60']

# Interface do Streamlit (entradas do usuário)
UF = st.selectbox('Unidade Federativa (UF)', ufs, index=ufs.index('SP') if 'SP' in ufs else 0)
ESCOLARIDADE = st.selectbox('Escolaridade', escolaridades, index=escolaridades.index('Superior Completo') if 'Superior Completo' in escolaridades else 0)
ESTADO_CIVIL = st.selectbox('Estado Civil', estados_civis, index=estados_civis.index('Solteiro') if 'Solteiro' in estados_civis else 0)
QT_FILHOS = st.number_input('Quantidade de Filhos', min_value=0, value=1)
CASA_PROPRIA = st.radio('Possui Casa Própria?', options=['Sim', 'Não'], index=0) # Default 'Sim'
QT_IMOVEIS = st.number_input('Quantidade de Imóveis', min_value=0, value=1)
VL_IMOVEIS = st.number_input('Valor dos Imóveis (R$)', min_value=0.0, value=300000.0, step=1000.0)
OUTRA_RENDA = st.radio('Possui outra renda?', options=['Sim', 'Não'], index=1) # Default 'Não'
if OUTRA_RENDA == 'Sim':
    OUTRA_RENDA_VALOR = st.number_input('Valor da Outra Renda (R$)', min_value=0.0, value=3000.0, step=100.0)
else:
    OUTRA_RENDA_VALOR = 0.0
TEMPO_ULTIMO_EMPREGO_MESES = st.select_slider('Tempo do Último Emprego (meses)', options=range(0, 241), value=18)
TRABALHANDO_ATUALMENTE = st.checkbox('Está trabalhando atualmente?', value=True)
ULTIMO_SALARIO = st.number_input('Último Salário (R$)', min_value=0.0, value=20400.0, step=100.0)
QT_CARROS_input = st.multiselect('Quantidade de Carros', [0, 1, 2, 3, 4, 5], default=[1])
VALOR_TABELA_CARROS = st.select_slider('Valor Tabela dos Carros (R$)', options=range(0, 200001, 5000), value=60000)
FAIXA_ETARIA = st.radio('Faixa Etária', faixas_etarias, index=faixas_etarias.index('36-45') if '36-45' in faixas_etarias else 0)


if st.button('Verificar Crédito'):
    # Converter as opções categóricas para valores numéricos
    uf_map = {label: i for i, label in enumerate(ufs)}
    escolaridade_map = {label: i for i, label in enumerate(escolaridades)}
    estado_civil_map = {label: i for i, label in enumerate(estados_civis)}
    faixa_etaria_map = {label: i for i, label in enumerate(faixas_etarias)}

    # Montar os novos dados na ordem correta de feature_names
    novos_dados_list = [
        uf_map[UF], escolaridade_map[ESCOLARIDADE], estado_civil_map[ESTADO_CIVIL], QT_FILHOS,
        1 if CASA_PROPRIA == 'Sim' else 0, QT_IMOVEIS, VL_IMOVEIS,
        1 if OUTRA_RENDA == 'Sim' else 0, OUTRA_RENDA_VALOR, TEMPO_ULTIMO_EMPREGO_MESES,
        1 if TRABALHANDO_ATUALMENTE else 0, ULTIMO_SALARIO, len(QT_CARROS_input), VALOR_TABELA_CARROS,
        faixa_etaria_map[FAIXA_ETARIA]
    ]

    # Transformar os dados para DataFrame com nomes de colunas para o scaler
    X_input_df = pd.DataFrame([novos_dados_list], columns=feature_names)

    try:
        # Aplicar o scaler (que espera nomes de features se foi treinado com eles)
        X_input_scaled_array = scaler.transform(X_input_df)
        # Converter o array escalado de volta para DataFrame para SHAP (alguns explainers e plots preferem/requerem)
        X_input_scaled_df = pd.DataFrame(X_input_scaled_array, columns=feature_names)

        # Fazer previsões
        score_previsto = lr_model.predict(X_input_scaled_array) # Modelo geralmente espera array numpy
        probabilidade_prevista = lr_model.predict_proba(X_input_scaled_array)

    except Exception as e:
        st.error(f"Erro durante a transformação dos dados ou predição: {e}")
        st.stop()

    # Exibir previsões
    st.subheader("Resultado da Análise de Crédito")
    predicao_classe = score_previsto[0]
    prob_classe_aprovado = probabilidade_prevista[0][1] # Probabilidade da classe 1 (Aprovado)

    resultado_texto = 'Aprovado !' if predicao_classe == 1 else 'Recusado !'
    cor_resultado = "green" if predicao_classe == 1 else "red"

    st.markdown(f"Seu crédito foi: <span style='color:{cor_resultado}; font-weight:bold;'>{resultado_texto}</span>", unsafe_allow_html=True)
    st.write(f"Probabilidade de Aprovação (classe 1): {prob_classe_aprovado:.2%}")


    # --- Explicações ---
    st.subheader("Entendendo a Decisão")
    
    # Determinar o tipo de modelo para SHAP (simplificado)
    # Idealmente, inspecione type(lr_model) ou tenha essa informação de outra forma.
    model_name_for_shap = "XGBoost" # Assumindo que lr_model é XGBoost ou compatível com TreeExplainer
    # if isinstance(lr_model, LogisticRegression): model_name_for_shap = "Logistic Regression"

    # --- SHAP ---
    if model_name_for_shap not in ['MLP', 'Naive Bayes', 'KNN']:
        try:
            st.markdown("---")
            st.markdown("#### Explicação com SHAP (Impacto das Features na Predição Atual)")
            
            if model_name_for_shap == "Logistic Regression":
                # Para Logistic Regression, X_train_scaled_df (dados de treino escalados) é o background data
                # explainer_shap = shap.LinearExplainer(lr_model, X_train_scaled_df)
                # shap_values_instance_array = explainer_shap.shap_values(X_input_scaled_df)
                # contribs_shap = shap_values_instance_array[0] # SHAP values para a instância atual
                st.info("SHAP para Regressão Logística ainda não implementado em detalhe nesta versão.")
                contribs_shap = None

            elif model_name_for_shap == "XGBoost": # Ou qualquer modelo baseado em árvore compatível
                explainer_shap = shap.TreeExplainer(lr_model)
                shap_values_instance_obj = explainer_shap(X_input_scaled_df) # Retorna objeto Explanation para a instância
                
                # Waterfall plot para a instância atual
                # O plot mostrará a contribuição para a saída da classe 1 (Aprovado)
                fig_waterfall, ax_waterfall = plt.subplots()
                shap.plots.waterfall(shap_values_instance_obj[0], show=False, max_display=10)
                ax_waterfall.set_title("Contribuição das Features para a Predição (SHAP)")
                st.pyplot(fig_waterfall)
                plt.clf() # Limpar a figura para evitar sobreposição em execuções futuras

                contribs_shap = shap_values_instance_obj.values[0] # Valores SHAP para a instância
            else:
                st.info(f"SHAP não configurado para o tipo de modelo: {model_name_for_shap}")
                contribs_shap = None

            if contribs_shap is not None:
                # Ordenar features pela magnitude da contribuição SHAP para a classe predita
                # Se predito Aprovado (1), queremos os SHAP values positivos.
                # Se predito Recusado (0), os SHAP values da classe 1 serão negativos.
                if predicao_classe == 0: # Recusado
                    st.write("**Principais fatores que influenciaram a recusa (contribuindo negativamente para 'Aprovado'):**")
                    sorted_indices_shap = np.argsort(contribs_shap) # Menor (mais negativo) para maior
                    top_contributors_indices = sorted_indices_shap[:3] # 3 mais negativos
                else: # Aprovado
                    st.write("**Principais fatores que influenciaram a aprovação (contribuindo positivamente para 'Aprovado'):**")
                    sorted_indices_shap = np.argsort(contribs_shap)
                    top_contributors_indices = sorted_indices_shap[:-4:-1] # 3 mais positivos
                
                razoes_shap_list = [f"{feature_names[j]} (contribuição SHAP: {contribs_shap[j]:.2f})" for j in top_contributors_indices]
                for razao in razoes_shap_list:
                    st.markdown(f"- {razao}")
                
                if predicao_classe == 0:
                    texto_exp_shap = f"A análise sugere que melhorias nos aspectos de '{', '.join([feature_names[j] for j in top_contributors_indices[:2]])}' e outros fatores desfavoráveis poderiam aumentar as chances de aprovação."
                    st.info(texto_exp_shap)

        except Exception as e:
            st.warning(f"Não foi possível gerar a explicação SHAP detalhada: {e}")
            # st.exception(e) # Descomente para debugging para ver o traceback completo no app

    else:
        st.info(f"Explicações SHAP não estão configuradas para o modelo {model_name_for_shap} neste aplicativo.")

    # --- Anchor ---
    try:
        st.markdown("---")
        st.markdown("#### Explicação com Anchor (Regra Mínima para a Decisão)")

        # predict_fn para Anchor: recebe array numpy 2D, retorna predições do modelo (0 ou 1)
        def predict_fn_anchor(data_anchor_array_2d):
            # Converter array numpy para DataFrame com colunas para o scaler
            data_anchor_df = pd.DataFrame(data_anchor_array_2d, columns=feature_names)
            data_anchor_scaled = scaler.transform(data_anchor_df)
            return lr_model.predict(data_anchor_scaled)

        # Nomes das classes como o Anchor espera (correspondendo à saída de predict_fn_anchor)
        class_names_anchor = ['Recusado', 'Aprovado']

        # X_train_df.values são os dados de treino NÃO escalados (como array NumPy)
        anchor_explainer_obj = anchor_tabular.AnchorTabularExplainer(
            class_names_anchor,
            feature_names,
            X_train_df.values # Anchor espera os dados de treino não escalados
        )
        
        # X_input_df.values[0] é a instância atual NÃO escalada como um array 1D
        anchor_exp = anchor_explainer_obj.explain_instance(
            X_input_df.values[0], # Passa a primeira (e única) linha como um array 1D
            predict_fn_anchor,
            threshold=0.95 # Limiar de precisão desejado para a regra do Anchor
        )

        st.write(f"**Regra Mínima Identificada (Anchor):** Se {' E '.join(anchor_exp.names())}, então a predição é **{class_names_anchor[predicao_classe]}**.")
        st.write(f"(Esta regra se aplica com precisão de {anchor_exp.precision():.2f} e tem uma cobertura de {anchor_exp.coverage():.2f} em casos similares).")
        
        if not anchor_exp.names():
             st.warning("O Anchor não conseguiu encontrar uma regra com o limiar de precisão de 0.95. Tente um limiar menor ou verifique os dados de treino.")

    except NameError: # Se anchor_tabular não foi importado com sucesso
        st.warning("A biblioteca Anchor não está disponível ou não foi importada corretamente. A explicação Anchor não pode ser gerada.")
    except Exception as e:
        st.warning(f"Não foi possível gerar a explicação Anchor: {e}")
        # st.exception(e) # Descomente para debugging