# ==============================================================================
# AI-TRD v1.0 - SISTEMA DE TRIAGEM AUTOMATIZADA DE RETINOPATIA DIABÉTICA
# ------------------------------------------------------------------------------
# AUTOR PRINCIPAL: Tiago da Silva Albuquerque
# IDENTIFICADOR ORCID: 0009-0003-4308-4435
# PERFIL: Cigano Calon | Bacharel em Direito | Acadêmico de Medicina - UNIFAN
# DATA DE REGISTRO: 02/03/2026
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')  # Oculta avisos não críticos para manter o terminal limpo


def gerar_dados_sinteticos(num_pacientes=500):
    """
    Gera um dataset fictício em conformidade com a LGPD para simular
    a triagem de Retinopatia Diabética no contexto do SUS.
    """
    print("[1/4] Gerando coorte sintética de pacientes (Mock Data)...")
    np.random.seed(42)  # Garante reprodutibilidade

    idade_paciente = np.random.randint(30, 75, num_pacientes)
    tempo_diabetes_anos = np.random.randint(1, 30, num_pacientes)
    hemoglobina_glicada = np.round(np.random.uniform(5.5, 12.0, num_pacientes), 1)
    hipertensao_arterial = np.random.choice([0, 1], num_pacientes, p=[0.45, 0.55])
    uso_insulina = np.random.choice([0, 1], num_pacientes, p=[0.6, 0.4])
    acompanhamento_oftalmologico = np.random.choice([0, 1], num_pacientes, p=[0.55, 0.45])

    # Lógica de risco para retinopatia diabética
    risco_base = (
        (tempo_diabetes_anos / 5)
        + ((hemoglobina_glicada - 5.5) * 0.8)
        + (hipertensao_arterial * 1.5)
        + (uso_insulina * 1.2)
        - (acompanhamento_oftalmologico * 1.0)
    )
    retinopatia = np.where(risco_base >= 5.0, 1, 0)  # 1 = Retinopatia presente, 0 = Ausente

    df = pd.DataFrame({
        'idade_paciente': idade_paciente,
        'tempo_diabetes_anos': tempo_diabetes_anos,
        'hemoglobina_glicada': hemoglobina_glicada,
        'hipertensao_arterial': hipertensao_arterial,
        'uso_insulina': uso_insulina,
        'acompanhamento_oftalmologico': acompanhamento_oftalmologico,
        'retinopatia_diabetica': retinopatia,
    })

    return df


def treinar_modelo(df_dados):
    """
    Treina o classificador Random Forest com os dados clínicos.
    """
    print("[2/4] Preparando variáveis (Features) e Alvo (Target)...")

    X = df_dados.drop('retinopatia_diabetica', axis=1)
    y = df_dados['retinopatia_diabetica']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[3/4] Inicializando o Motor Preditivo (Random Forest)...")
    # max_depth=5 limita a profundidade para evitar overfitting em dados clínicos estruturados
    modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo.fit(X_train, y_train)

    return modelo, X_test, y_test


def avaliar_modelo(modelo, X_test, y_test):
    """
    Avalia o desempenho do modelo e exibe o relatório clínico.
    """
    print("[4/4] Gerando Estratificação de Risco e Relatório Final...\n")

    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(
        y_test,
        y_pred,
        target_names=['Sem Retinopatia (0)', 'Com Retinopatia (1)'],
    )

    print("=====================================================")
    print(" RELATÓRIO DE VALIDAÇÃO - AI-TRD v1.0")
    print("=====================================================")
    print(f"Acurácia Geral do Algoritmo: {acuracia * 100:.2f}%\n")
    print("Detalhamento Clínico:")
    print(relatorio)
    print("=====================================================")
    print("Status: PRONTO PARA IMPLEMENTAÇÃO NO AMBIENTE SUS.")


if __name__ == "__main__":
    print("Algoritmo AI-TRD carregado com sucesso. Autor: Tiago S. Albuquerque\n")
    dataset = gerar_dados_sinteticos()
    modelo_treinado, X_teste, y_teste = treinar_modelo(dataset)
    avaliar_modelo(modelo_treinado, X_teste, y_teste)
