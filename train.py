import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

# =============================
# 1. LOAD DATASET
# =============================
df = pd.read_csv("dataset_sampah_final2020-2023.csv", sep=";", decimal=",")

# normalisasi kolom
df.columns = df.columns.str.strip().str.lower()

# =============================
# 2. CLEANING & KONVERSI NUMERIC
# =============================
text_cols = ["kabupaten"]

for c in df.columns:
    if c not in text_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.fillna(0)

# =============================
# 3. AMBIL FITUR
# =============================
features = [
    "tahun",
    "jumlah_penduduk",
    "total_sampah_ton",
    "jumlah_truk",
    "jumlah_motor",
    "jumlah_tps",
    "total_armada",
    "sampah_perpenduduk",
    "sampah_perarmada",
    "sampah_pertps"
]

X = df[features]

# =============================
# 4. SCALING
# =============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================
# 5. TRAIN KMEANS
# =============================
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters

# =============================
# 6. TRAIN PCA (2 KOMPONEN)
# =============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["pc1"] = X_pca[:, 0]
df["pc2"] = X_pca[:, 1]

# =============================
# 7. SIMPAN MODEL
# =============================
pickle.dump(kmeans, open("model_kmeans.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(pca, open("pca.pkl", "wb"))

# =============================
# 8. SIMPAN DATASET FINAL
# =============================
df.to_csv("dataset_cluster_final.csv", sep=";", index=False)

print("TRAINING SELESAI!")
print("File berhasil dibuat:")
print("- model_kmeans.pkl")
print("- scaler.pkl")
print("- pca.pkl")
print("- dataset_cluster_final.csv")
