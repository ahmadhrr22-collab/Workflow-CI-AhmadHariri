import pandas as pd
from sklearn.cluster import KMeans
import mlflow
import mlflow.sklearn

# --- KONFIGURASI ---
DATA_PATH = 'preprocessed_data.csv' 

def train_model():
    print("Memuat data...")
    try:
        data = pd.read_csv(DATA_PATH)
        print(f"Data berhasil dimuat: {data.shape}")
    except FileNotFoundError:
        print(f"Error: File '{DATA_PATH}' tidak ditemukan. Pastikan lokasinya benar.")
        return

    # 1. AKTIFKAN AUTOLOG
    mlflow.sklearn.autolog()

    # 2. MULAI EXPERIMENT
    
    with mlflow.start_run(run_name="Basic_Autolog_Run"):
        print("Memulai Training KMeans dengan Autolog...")
        
        # Inisialisasi Model (Contoh K=3, sesuaikan jika perlu)
        kmeans = KMeans(n_clusters=3, random_state=42)
        
        # Train Model
        # Saat .fit() dipanggil, autolog akan bekerja otomatis
        kmeans.fit(data)
        
        print("Training selesai!")

if __name__ == "__main__":
    train_model()