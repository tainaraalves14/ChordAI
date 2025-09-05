import os
import librosa
import numpy as np
import pandas as pd

# Caminho absoluto da pasta onde o script está
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho da pasta 'data' (uma pasta acima do script)
AUDIO_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))

# Caminho do CSV de saída
OUTPUT_CSV = os.path.abspath(os.path.join(BASE_DIR, '..', 'processed_features.csv'))

def extract_features(file_path):
    try:
        # Carregar áudio com dtype=float32 explicitamente
        y, sr = librosa.load(file_path, mono=True, duration=30, dtype=np.float32)
        
        # Extrair MFCCs e garantir float32
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).astype(np.float32)
        mfccs_mean = np.mean(mfccs.T, axis=0, dtype=np.float32)
        
        # Verificar tipo de dados
        if mfccs_mean.dtype != np.float32:
            mfccs_mean = mfccs_mean.astype(np.float32)
        
        return mfccs_mean
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None

def process_all_songs():
    features_list = []

    # Certifica que a pasta existe
    if not os.path.exists(AUDIO_DIR):
        print(f"Pasta de áudio não encontrada: {AUDIO_DIR}")
        return

    for filename in os.listdir(AUDIO_DIR):
        if filename.endswith(('.mp3', '.wav', '.flac')):
            file_path = os.path.join(AUDIO_DIR, filename)
            song_features = extract_features(file_path)

            if song_features is not None:
                features_dict = {'filename': filename}
                for i, feature in enumerate(song_features):
                    features_dict[f'mfcc_{i+1}'] = feature
                features_list.append(features_dict)

    if features_list:
        features_df = pd.DataFrame(features_list)
        # Garantir que as colunas numéricas sejam float32
        for col in features_df.columns:
            if col.startswith('mfcc_'):
                features_df[col] = features_df[col].astype(np.float32)
        features_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Características salvas em {OUTPUT_CSV}")
    else:
        print("Nenhum arquivo de áudio processado.")

if __name__ == "__main__":
    process_all_songs()