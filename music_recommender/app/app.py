import streamlit as st
import pandas as pd
import joblib
import os
import librosa
import numpy as np
import tempfile

# ----- Caminhos absolutos para os arquivos -----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sobe de app/ para music_recommender
FEATURES_CSV = os.path.join(BASE_DIR, "processed_features.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "kmeans_model.joblib")

# ----- Verificação da existência dos arquivos -----
if not os.path.exists(FEATURES_CSV) or not os.path.exists(MODEL_FILE):
    st.error(
        "Arquivos de dados ou modelo não encontrados.\n"
        "Por favor, execute os scripts process_audio.py e train_model.py primeiro."
    )
    st.stop()  # para não continuar se os arquivos não existirem

# Carregar CSV e modelo
df = pd.read_csv(FEATURES_CSV)
kmeans = joblib.load(MODEL_FILE)

# ----- Função utilitária para garantir float32 e 2D -----
def prepare_features(features):
    """Converte qualquer array para float32 e garante 2D para o KMeans."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    return features

# ----- Função para extrair features do upload -----
def extract_features_from_uploaded_file(uploaded_file):
    """Extrai características de um arquivo de áudio carregado pelo Streamlit."""
    tmp_path = None
    try:
        # Salva temporariamente o arquivo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Carregar áudio com tipo float32 explicitamente
        y, sr = librosa.load(tmp_path, mono=True, duration=30, dtype=np.float32)
        
        # Extrair MFCCs e garantir float32
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).astype(np.float32)
        mfccs_mean = np.mean(mfccs.T, axis=0, dtype=np.float32)
        
        # Verificar tipo de dados
        if mfccs_mean.dtype != np.float32:
            mfccs_mean = mfccs_mean.astype(np.float32)
        
        return prepare_features(mfccs_mean)
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ----- Função para gerar recomendações -----
def get_recommendations(input_features, num_recommendations=5):
    """Gera recomendações com base nas features da música de entrada."""
    input_features = prepare_features(input_features)

    try:
        input_cluster = kmeans.predict(input_features)[0]
    except Exception as e:
        st.error(f"Erro ao classificar o áudio: {e}")
        return []

    # Seleciona músicas do mesmo cluster
    same_cluster_songs = df[df['cluster'] == input_cluster].copy()

    # Para evitar recomendar o próprio arquivo, se houver
    input_filename = "Uploaded File"
    if input_filename in same_cluster_songs['filename'].values:
        same_cluster_songs = same_cluster_songs[same_cluster_songs['filename'] != input_filename]

    if same_cluster_songs.empty:
        return "Nenhuma recomendação encontrada neste cluster."

    recommendations = same_cluster_songs['filename'].sample(
        n=min(num_recommendations, len(same_cluster_songs)),
        random_state=42  # garante reprodutibilidade
    )
    return recommendations.tolist()

# ----- Interface Streamlit -----
st.title("Sistema de Recomendação de Músicas com IA")
st.markdown("Faça o upload de uma música para receber recomendações baseadas nas suas características sonoras!")

uploaded_file = st.file_uploader("Escolha um arquivo de áudio (MP3, WAV, FLAC)", type=["mp3", "wav", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')

    if st.button("Gerar Recomendações"):
        with st.spinner("Processando áudio e gerando recomendações..."):
            input_features = extract_features_from_uploaded_file(uploaded_file)

            if input_features is not None:
                # Verificar tipo de dados antes da predição
                if input_features.dtype != np.float32:
                    input_features = input_features.astype(np.float32)
                
                # Exibe o cluster previsto
                try:
                    input_cluster = kmeans.predict(input_features)[0]
                    st.info(f"O áudio enviado foi classificado no cluster **#{input_cluster}**.")
                except Exception as e:
                    st.error(f"Erro ao prever cluster: {e}")
                    st.stop()

                # Gera e exibe recomendações
                recommendations = get_recommendations(input_features)
                
                st.subheader("Aqui estão suas recomendações:")
                if isinstance(recommendations, str):
                    st.write(recommendations)
                else:
                    for song in recommendations:
                        st.write(f"- {song}")