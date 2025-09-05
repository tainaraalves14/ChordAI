# Passo 1: Importar tudo o que precisamos
# As mesmas bibliotecas que usamos antes, mais o 'os' para navegar nas pastas
import pandas as pd
import joblib 
import os
import librosa
import numpy as np

# Passo 2: Definir os caminhos para os arquivos
FEATURES_CSV = '../processed_features.csv'
MODEL_FILE = '../models/kmeans_model.joblib'
AUDIO_DIR = '../data'

# Passo 3: Recriar a função de extração de features
# A função é a mesma, mas a copiamos aqui para que este script seja autônomo.
def extract_features(file_path):
    """Extrai as características da música de entrada para o modelo entender."""
    try:
        y, sr = librosa.load(file_path, mono=True, duration=30)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # O .reshape(1, -1) é crucial! Ele transforma o vetor em uma matriz de uma linha,
        # que é o formato que o método .predict() do scikit-learn espera.
        return mfccs_mean.reshape(1, -1)
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None
    
    # Passo 4: Criar a função principal de recomendação
def get_recommendations(input_song_filename, num_recommendations=5):
    """
    Carrega o modelo treinado, encontra o cluster da música de entrada e
    retorna uma lista de músicas do mesmo cluster.
    """
    # 1. Carregar o modelo treinado e o DataFrame com os clusters
    kmeans = joblib.load(MODEL_FILE)
    df = pd.read_csv(FEATURES_CSV)
    
    # 2. Processar a música de entrada
    file_path = os.path.join(AUDIO_DIR, input_song_filename)
    input_features = extract_features(file_path)
    
    if input_features is None:
        return "Erro: Não foi possível processar a música de entrada."

    # 3. Prever o cluster da música de entrada
    # O .predict() usa o conhecimento do modelo para dizer a qual grupo a música pertence.
    input_cluster = kmeans.predict(input_features)[0]
    print(f"A música '{input_song_filename}' pertence ao cluster {input_cluster}")
    
    # 4. Encontrar as músicas no mesmo cluster
    # Selecionamos todas as linhas (músicas) do DataFrame que têm o mesmo número de cluster.
    same_cluster_songs = df[df['cluster'] == input_cluster]
    
    # 5. Remover a própria música da lista de recomendações
    same_cluster_songs = same_cluster_songs[same_cluster_songs['filename'] != input_song_filename]
    
    if same_cluster_songs.empty:
        return "Nenhuma recomendação encontrada neste cluster."
    
    # 6. Selecionar e retornar as recomendações
    # O .sample() garante que as recomendações sejam aleatórias
    recommendations = same_cluster_songs['filename'].sample(n=min(num_recommendations, len(same_cluster_songs)))
    
    return recommendations.tolist()

    # Passo 5: Testar a lógica de recomendação
    if __name__ == "__main__":
        # Altere o nome do arquivo para uma música que você tenha no seu dataset
        song_to_recommend = "nome_da_sua_musica.mp3" 
        
        if os.path.exists(os.path.join(AUDIO_DIR, song_to_recommend)):
            recommended_songs = get_recommendations(song_to_recommend)
            print("\nRecomendações para '{song_to_recommend}':")
            for song in recommended_songs:
                print(f"- {song}")
        else:
            print(f"Erro: Arquivo '{song_to_recommend}' não encontrado na pasta de dados.")