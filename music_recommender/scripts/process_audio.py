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
        y, sr = librosa.load(file_path, mono=True, duration=30)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
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
        features_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Características salvas em {OUTPUT_CSV}")
    else:
        print("Nenhum arquivo de áudio processado.")

if __name__ == "__main__":
    process_all_songs()
