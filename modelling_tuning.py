# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from yellowbrick.cluster import KElbowVisualizer
import joblib
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn

# ========================
# 1. LOAD ENV VARIABLES
# ========================
load_dotenv() # Membaca file .env

print("Melakukan konfigurasi Environment...")

# Set environment variables untuk autentikasi 
required_vars = [
    "MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"
]

for var in required_vars:
    if not os.getenv(var):
        print(f"WARNING: Variabel {var} tidak ditemukan di .env!")

# ========================
# 2. MLflow CONFIGURATION
# ========================
try:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("clustering-experiment")
    print("Koneksi MLflow berhasil diinisialisasi.")
except Exception as e:
    print(f"Error setting MLflow: {e}")

# ========================
# 3. ML TRAINING PIPELINE
# ========================
with mlflow.start_run():
    print("\n=== MULAI EXPERIMENT ===")

    # -----------------------------------------------------------
    # A. Load & Clean Data (REVISI - SMART LOADING)
    # -----------------------------------------------------------
    csv_filename = "preprocessed_data.csv"
    
    # Cari lokasi file
    if os.path.exists(csv_filename):
        data_path = csv_filename
    elif os.path.exists(os.path.join("namadataset_preprocessing", csv_filename)):
        data_path = os.path.join("namadataset_preprocessing", csv_filename)
    else:
        print(f"ERROR: File {csv_filename} tidak ditemukan di folder manapun!")
        exit()

    print(f"Membaca data dari: {data_path}")
    
    # --- SMART LOAD: Deteksi otomatis separator ---
    try:
        # engine='python' dengan sep=None akan mendeteksi otomatis (koma atau titik koma)
        df = pd.read_csv(data_path, sep=None, engine='python')
    except Exception as e:
        print(f"Gagal Smart Load: {e}")
        # Fallback manual
        try:
            df = pd.read_csv(data_path, sep=',')
            if df.shape[1] < 2: 
                df = pd.read_csv(data_path, sep=';')
        except:
            print("Gagal membaca file. Pastikan format CSV benar.")
            exit()

    print(f"Data Awal (Shape): {df.shape}")
    print("Contoh Data (Head):")
    print(df.head(2))

    # --- PAKSA KONVERSI KE ANGKA ---
    # Kadang angka terbaca sebagai string karena ada 1 nilai aneh. Kita paksa ubah.
    # Kolom teks murni (seperti Nama/ID) akan berubah jadi NaN (kosong) dan kita buang.
    
    # Simpan nama kolom asli
    original_cols = df.columns.tolist()
    
    # Buat dataframe baru khusus angka
    df_numeric = df.copy()
    
    for col in df_numeric.columns:
        # Coba ubah setiap kolom jadi angka. Jika gagal (teks), ubah jadi NaN
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    # Buang kolom yang isinya NaN semua (berarti itu kolom teks asli seperti ID/Nama)
    df_numeric = df_numeric.dropna(axis=1, how='all')
    
    # Buang baris yang masih ada NaN (sisa kotoran)
    df_numeric = df_numeric.dropna()

    print(f"Data Numerik (siap training): {df_numeric.shape}")
    
    if df_numeric.shape[1] == 0:
        print("ERROR FATAL: Semua kolom dianggap teks.")
        print("Cek file CSV Anda. Apakah isinya teks semua? Atau format angkanya salah (misal pakai koma 14,5 bukan titik 14.5)?")
        exit()
        
    print("Kolom yang digunakan:", df_numeric.columns.tolist())

    # -----------------------------------------------------------
    # B. Elbow Method
    # -----------------------------------------------------------
    print("\n--- Menjalankan Elbow Method ---")
    model_elbow = KMeans(random_state=42)
    visualizer = KElbowVisualizer(
        model_elbow, 
        k=(2, 10), 
        metric="silhouette", 
        timings=False
    )
    
    visualizer.fit(df_numeric) # Fit pada data numerik!
    
    # Simpan plot ke file gambar agar bisa di-upload
    elbow_plot_name = "elbow_method.png"
    visualizer.show(outpath=elbow_plot_name) 
    mlflow.log_artifact(elbow_plot_name) # Upload ke DagsHub
    print(f"Grafik Elbow disimpan & diupload: {elbow_plot_name}")

    best_k_elbow = visualizer.elbow_value_
    mlflow.log_param("elbow_best_k", best_k_elbow)
    print("Best k from Elbow:", best_k_elbow)
    plt.clf() # Bersihkan plot agar tidak tumpang tindih

    # -----------------------------------------------------------
    # C. Hyperparameter Tuning (Manual Grid Search)
    # -----------------------------------------------------------
    print("\n--- Menjalankan Hyperparameter Tuning ---")
    
    param_grid = {
        "n_clusters": list(range(2, 6)), # Diperkecil range-nya agar cepat (bisa diubah)
        "init": ["k-means++"],
        "n_init": [10],
        "max_iter": [300]
    }

    best_score = -1
    best_params = None
    results = []

    # Loop Tuning
    for n_clusters in param_grid["n_clusters"]:
        for init in param_grid["init"]:
            for n_init in param_grid["n_init"]:
                for max_iter in param_grid["max_iter"]:
                    
                    # Train model sementara
                    model = KMeans(
                        n_clusters=n_clusters,
                        init=init,
                        n_init=n_init,
                        max_iter=max_iter,
                        random_state=42
                    )
                    
                    # Fit pada data numerik!
                    labels = model.fit_predict(df_numeric)
                    
                    # Hitung score
                    try:
                        score = silhouette_score(df_numeric, labels)
                    except:
                        score = -1 # Handle jika cluster cuma 1 (error silhouette)

                    results.append({
                        "n_clusters": n_clusters,
                        "silhouette_score": score,
                        "params": str(model.get_params())
                    })

                    if score > best_score:
                        best_score = score
                        best_params = {
                            "n_clusters": n_clusters,
                            "init": init,
                            "n_init": n_init,
                            "max_iter": max_iter
                        }

    # Simpan hasil tuning
    results_df = pd.DataFrame(results)
    results_df.to_csv("tuning_results.csv", index=False)
    mlflow.log_artifact("tuning_results.csv")
    
    print(f"Best Silhouette Score: {best_score}")
    print(f"Best Params: {best_params}")

    # Log parameter terbaik ke MLflow
    if best_params:
        for key, value in best_params.items():
            mlflow.log_param(f"best_{key}", value)
    else:
        # Fallback jika tuning gagal
        best_params = {"n_clusters": 3, "random_state": 42}

    # -----------------------------------------------------------
    # D. Train Best Model & Evaluation
    # -----------------------------------------------------------
    print("\n--- Training Model Terbaik ---")
    
    best_model = KMeans(**best_params, random_state=42)
    best_model.fit(df_numeric) # Fit numeric
    labels = best_model.labels_

    # Metrics
    sil_score = silhouette_score(df_numeric, labels)
    ch_score = calinski_harabasz_score(df_numeric, labels)
    db_score = davies_bouldin_score(df_numeric, labels)
    inertia = best_model.inertia_

    # Log Metrics
    mlflow.log_metric("silhouette_score", sil_score)
    mlflow.log_metric("calinski_harabasz", ch_score)
    mlflow.log_metric("davies_bouldin", db_score)
    mlflow.log_metric("inertia", inertia)

    print(f"Silhouette: {sil_score:.4f}")
    print(f"Davies-Bouldin: {db_score:.4f}")

    # -----------------------------------------------------------
    # E. Save Model (Log for Serving)
    # -----------------------------------------------------------
    print("\n--- Menyimpan Model ---")

    # 1. Simpan Lokal (Backup)
    os.makedirs("artifacts", exist_ok=True)
    local_model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(best_model, local_model_path)
    
    # 2. Log Model ke MLflow (Standard Format untuk Serving)
    # Ini akan membuat folder 'model' di DagsHub yang berisi file yang dibutuhkan Kriteria 4
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="KMeans_Best_Model"
    )
    print("Model berhasil disimpan ke DagsHub dengan format MLmodel.")
    
    print("\n=== SELESAI! Cek Dashboard DagsHub Anda ===")