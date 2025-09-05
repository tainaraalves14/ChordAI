import os
import pandas as pd
from sklearn.cluster import KMeans
import joblib
import argparse
import numpy as np

# -----------------------------
# 1️⃣ Caminhos absolutos
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES_CSV = os.path.join(BASE_DIR, "../processed_features.csv")
MODEL_FILE = os.path.join(BASE_DIR, "../models/kmeans_model.joblib")

# Cria a pasta do modelo se não existir
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

# -----------------------------
# 2️⃣ Função para treinar o modelo
# -----------------------------
def train_model(n_clusters=10):
    # Verifica se o CSV existe
    if not os.path.exists(FEATURES_CSV):
        print(f"CSV de features não encontrado: {FEATURES_CSV}")
        return

    # Carrega os dados
    df = pd.read_csv(FEATURES_CSV)
    
    # Separar as features (excluindo 'filename')
    if 'filename' in df.columns:
        features = df.drop('filename', axis=1)
    else:
        features = df.copy()

    # Garantir que as features sejam float32
    features = features.astype(np.float32)
    
    print(f"Treinando o modelo K-Means com {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features)

    # Adiciona os clusters ao DataFrame
    df['cluster'] = kmeans.labels_
    
    # Garantir que a coluna 'cluster' seja int32 para consistência
    df['cluster'] = df['cluster'].astype(np.int32)
    
    # Garantir que as colunas de features no DataFrame sejam float32 antes de salvar
    for col in df.columns:
        if col.startswith('mfcc_'):
            df[col] = df[col].astype(np.float32)
    
    df.to_csv(FEATURES_CSV, index=False)

    # Salva o modelo
    joblib.dump(kmeans, MODEL_FILE)

    print("✅ Modelo treinado e salvo com sucesso!")
    print(f"DataFrame com clusters salvo em: {FEATURES_CSV}")
    print(f"Modelo K-Means salvo em: {MODEL_FILE}")

# -----------------------------
# 3️⃣ Execução pelo terminal
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar K-Means com features de áudio")
    parser.add_argument("--clusters", type=int, default=10, help="Número de clusters")
    args = parser.parse_args()

    train_model(n_clusters=args.clusters)