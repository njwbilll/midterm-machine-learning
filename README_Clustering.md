# 🛒 Midterm Machine Learning — Customer Credit Card Clustering

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![KMeans](https://img.shields.io/badge/Model-KMeans-orange)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
![Colab](https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab)

---

## 👤 Identitas
**Nama**:  Najwa Bilqis Al Khalidah |
| **NIM**: 101032300186 |
| **Kelas**: TK-46-GAB 

---

## 📌 Tujuan Proyek

Membangun pipeline end-to-end **unsupervised machine learning (clustering)** untuk mengelompokkan pelanggan kartu kredit berdasarkan perilaku pengeluaran dan pembayaran mereka, menggunakan metode K-Means dan Agglomerative Clustering.

---

## 📂 Struktur Repository

```
midterm-machine-learning/
│
├── notebooks/
│   └── ML_CustomerClustering.ipynb   ← Main notebook
│
├── README.md                          ← Dokumentasi ini
└── requirements.txt                   ← Daftar library
```

---

## 🗂️ Dataset

**File:** `clusteringmidterm.csv`

| Info | Detail |
|------|--------|
| Jumlah data | 8.950 baris |
| Jumlah fitur | 18 kolom |
| Target | Tidak ada (unsupervised) |
| Missing values | `CREDIT_LIMIT`: 1, `MINIMUM_PAYMENTS`: 313 |

### Deskripsi Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| `BALANCE` | Saldo outstanding rata-rata pelanggan |
| `PURCHASES` | Total pembelian menggunakan kartu |
| `ONEOFF_PURCHASES` | Pembelian sekali bayar (lump sum) |
| `INSTALLMENTS_PURCHASES` | Pembelian dengan cicilan |
| `CASH_ADVANCE` | Total penarikan tunai |
| `CREDIT_LIMIT` | Batas kredit maksimum |
| `PAYMENTS` | Total pembayaran yang dilakukan |
| `MINIMUM_PAYMENTS` | Total minimum payment yang dilakukan |
| `PRC_FULL_PAYMENT` | Proporsi bulan di mana pelanggan membayar penuh |
| `TENURE` | Lama berlangganan (bulan) |

---

## ⚙️ Workflow Pipeline

```
Load Data
    ↓
EDA (distribusi fitur)
    ↓
Preprocessing
    ├── Imputasi missing values (median)
    ├── Handling outliers (IQR clipping)
    └── Feature engineering (rasio purchase/limit, payment/balance)
    ↓
StandardScaler
    ↓
Pilih jumlah cluster optimal
    ├── Elbow Method (Inertia)
    ├── Silhouette Score
    ├── Davies-Bouldin Score
    └── Calinski-Harabasz Score
    ↓
Model Training
    ├── KMeans Clustering
    └── Agglomerative Clustering (Ward)
    ↓
Visualisasi (PCA 2D)
    ↓
Interpretasi & Profil Cluster
    ↓
Tracking dengan MLflow
    ↓
Simpan Model & Hasil
```

---

## 🤖 Model yang Digunakan

### 1. KMeans Clustering ✅ (Model Terbaik)
- Algoritma clustering berbasis centroid
- Jumlah cluster dipilih berdasarkan **Silhouette Score tertinggi**
- Menggunakan `n_init=10` untuk stabilitas hasil
- Input: fitur yang sudah di-scale dengan `StandardScaler`

### 2. Agglomerative Clustering
- Algoritma hierarchical clustering (bottom-up)
- Linkage method: `ward` (meminimalkan varians dalam cluster)
- Digunakan sebagai pembanding terhadap KMeans

---

## 📊 Hasil Evaluasi

| Model | Silhouette Score ↑ | Davies-Bouldin Score ↓ | Calinski-Harabasz Score ↑ |
|-------|-------------------|------------------------|--------------------------|
| **KMeans** | **0.2830** | **1.6401** | **2227.86** |
| Agglomerative | 0.1833 | 1.9596 | 1833.90 |

> ↑ = higher is better, ↓ = lower is better

### Interpretasi Hasil
- **KMeans unggul di semua metrik** dibanding Agglomerative Clustering
- Silhouette Score 0.28 menunjukkan cluster yang cukup terpisah untuk data pelanggan yang memang memiliki perilaku beragam
- Davies-Bouldin 1.64 (KMeans) vs 1.96 (Agglomerative) → cluster KMeans lebih kompak dan terpisah
- KMeans dipilih sebagai **model final**

---

## 👥 Profil Cluster (KMeans — 2 Cluster)

| Fitur | Cluster 0 | Cluster 1 |
|-------|-----------|-----------|
| BALANCE | 1,928.65 | 1,445.85 |
| PURCHASES | **3,123.16** | 312.67 |
| ONEOFF_PURCHASES | **1,899.50** | 166.69 |
| INSTALLMENTS_PURCHASES | **1,223.97** | 146.28 |
| CASH_ADVANCE | 642.93 | **1,088.30** |
| PURCHASES_FREQUENCY | **0.91** | 0.35 |
| CASH_ADVANCE_FREQUENCY | 0.08 | **0.15** |
| CREDIT_LIMIT | **6,384.18** | 3,878.69 |
| PAYMENTS | **3,245.03** | 1,240.68 |
| PRC_FULL_PAYMENT | **0.27** | 0.12 |

### 🔵 Cluster 0 — "Active Spender" (Pelanggan Aktif Berbelanja)
Pelanggan dengan **pembelian sangat tinggi** (rata-rata Rp 3,123), credit limit besar (Rp 6,384), dan frekuensi belanja tinggi (0.91). Mereka aktif berbelanja baik secara lump sum maupun cicilan, jarang melakukan cash advance, dan cukup rajin membayar penuh. Segmen ini adalah **pelanggan premium yang profitable**.

### 🔴 Cluster 1 — "Cash Advance User" (Pengguna Tarik Tunai)
Pelanggan dengan **pembelian rendah** (rata-rata Rp 312) namun **cash advance tinggi** (Rp 1,088). Credit limit lebih kecil, pembayaran minimum rendah, dan jarang membayar penuh (PRC 0.12). Segmen ini cenderung **menggunakan kartu kredit sebagai sumber tunai darurat** dan berisiko lebih tinggi terhadap kredit macet.

---

## 📈 Tracking dengan MLflow

Semua eksperimen di-track menggunakan **MLflow** dengan nama experiment `CustomerClustering_ML`, meliputi:
- Parameter model (n_clusters, linkage)
- Metrik evaluasi (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Model artifact

---

## 🚀 Cara Menjalankan

1. **Buka notebook di Google Colab**

2. **Mount Google Drive & sesuaikan path dataset:**
   ```python
   BASE_PATH = '/content/drive/MyDrive/ML Dataset/'
   ```

3. **Jalankan semua cell secara berurutan**

4. **Install dependencies** (sudah ada di Cell 1):
   ```bash
   pip install optuna mlflow kneed
   ```

---

## 📁 Navigasi Notebook

| Cell | Section | Deskripsi |
|------|---------|-----------|
| 1 | Install & Import | Install library, import semua modul |
| 2 | Load & EDA | Load CSV, statistik deskriptif |
| 3 | Visualisasi Distribusi | Histogram semua fitur |
| 4 | Preprocessing | Imputasi, outlier clipping, feature engineering, scaling |
| 5 | Pilih K Optimal | Elbow method, Silhouette, Davies-Bouldin, Calinski-Harabasz |
| 6 | KMeans | Training KMeans + logging MLflow |
| 7 | Agglomerative | Training Agglomerative + logging MLflow |
| 8 | Visualisasi PCA | Scatter plot cluster dalam 2D |
| 9 | Profil Cluster | Heatmap & interpretasi tiap cluster |
| 10 | Perbandingan | Tabel & bar chart semua model |
| 11 | Simpan | Simpan model, scaler, PCA, dan hasil CSV ke Drive |

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
mlflow
kneed
joblib
```
