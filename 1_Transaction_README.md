# 🔍 Midterm Machine Learning — Online Transaction Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
![Optuna](https://img.shields.io/badge/Tuning-Optuna-purple)
![Colab](https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab)

---

## 👤 Identitas

| Field | Detail |
|-------|--------|
| **Nama** | Najwa Bilqis Al Khalidah |
| **NIM** | 101032300186 |
| **Kelas** | TK-46-GAB |

---

## 📌 Tujuan Proyek

Membangun pipeline end-to-end **machine learning klasifikasi** untuk memprediksi probabilitas sebuah transaksi online bersifat **fraudulent (penipuan)**, menggunakan dataset transaksi IEEE-CIS Fraud Detection.

---

## 📂 Struktur Repository

```
midterm-machine-learning/
│
├── notebooks/
│   └── ML_FraudDetection.ipynb   ← Main notebook (semua pipeline ada di sini)
│
├── README.md                      ← Dokumentasi ini
└── requirements.txt               ← Daftar library yang digunakan
```

---

## 🗂️ Dataset

**File:** `train_transaction.csv`

| Info | Detail |
|------|--------|
| Jumlah data | 590.540 baris |
| Jumlah fitur | 394 kolom |
| Target | `isFraud` — 1 (fraud) atau 0 (not fraud) |
| Fraud rate | ~3.50% → heavily imbalanced |
| Tipe fitur | Numerik & kategorikal (amount, time, product code, card info, address, anonymous V-features) |

> Dataset memiliki banyak missing values (beberapa kolom >90% kosong) dan class imbalance yang signifikan, sehingga membutuhkan penanganan khusus di preprocessing.

---

## ⚙️ Workflow Pipeline

```
Load Data
        ↓
Drop kolom missing > 90%
        ↓
Feature Engineering (log amount, hour, day, week)
        ↓
Label Encoding + Imputation
        ↓
Train-Validation Split (80/20, stratified)
        ↓
StandardScaler (untuk Logistic Regression)
        ↓
Model Training
    ├── Logistic Regression (Baseline)
    └── XGBoost + Optuna Hyperparameter Tuning (20 trials)
        ↓
Evaluasi (ROC-AUC, PR-AUC, Classification Report)
        ↓
Tracking dengan MLflow
        ↓
Simpan Model + Generate Predictions
```

---

## 🤖 Model yang Digunakan

### 1. Logistic Regression (Baseline)
- Model linear sederhana sebagai **baseline**
- Menggunakan `class_weight='balanced'` untuk menangani class imbalance
- Input: fitur yang sudah di-scale dengan `StandardScaler`

### 2. XGBoost Classifier (Tuned)
- Gradient boosting berbasis pohon keputusan
- Menggunakan `scale_pos_weight` untuk menangani class imbalance
- Hyperparameter dioptimasi menggunakan **Optuna** (20 trials, maximize ROC-AUC)
- Parameter yang di-tune: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`
- Menggunakan GPU (`device='cuda'`) bila tersedia, fallback ke CPU otomatis

---

## 📊 Hasil Evaluasi

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Logistic Regression | 0.8300 | 0.2982 |
| **XGBoost (Tuned)** | **0.9351** | **0.6190** |

### Interpretasi Hasil

- **XGBoost jauh mengungguli** Logistic Regression di kedua metrik
- **ROC-AUC 0.935** → model sangat baik dalam membedakan transaksi fraud vs non-fraud
- **PR-AUC digunakan sebagai metrik utama** karena dataset sangat imbalanced (3.5% fraud)
- **PR-AUC XGBoost 0.619** vs Logistic Regression 0.298 → XGBoost 2x lebih baik dalam mendeteksi fraud secara presisi
- Logistic Regression sebagai baseline sudah cukup baik (ROC-AUC 0.83) namun masih jauh tertinggal karena hubungan fitur-target yang tidak linear

---

## 📈 Tracking dengan MLflow

Semua eksperimen di-track menggunakan **MLflow** dengan nama experiment `FraudDetection_ML`, meliputi:
- Parameter model dan hyperparameter terbaik hasil Optuna
- Metrik evaluasi (ROC-AUC, PR-AUC) untuk setiap trial
- Model artifact (Logistic Regression & XGBoost final)

---

## 🚀 Cara Menjalankan

1. **Buka notebook di Google Colab**

2. **Aktifkan GPU (opsional, mempercepat XGBoost):**
   ```
   Runtime → Change runtime type → T4 GPU → Save
   ```

3. **Mount Google Drive & sesuaikan path dataset:**
   ```python
   BASE_PATH = '/content/drive/MyDrive/ML Dataset/'
   ```

4. **Jalankan semua cell secara berurutan** dari atas ke bawah

5. **Install dependencies** (sudah ada di Cell 1):
   ```bash
   pip install optuna mlflow
   ```

---

## 📁 Navigasi Notebook

| Cell | Section | Deskripsi |
|------|---------|-----------|
| 1 | Install & Import | Install library tambahan, import semua modul |
| 2 | Load Data | Load CSV dari Google Drive |
| 3 | EDA | Visualisasi distribusi class, amount, missing values |
| 4 | Preprocessing | Drop kolom missing >90%, feature engineering, encoding, imputation |
| 5 | Split & Scale | Train-val split stratified 80/20, StandardScaler |
| 6 | Evaluation Helper | Fungsi `evaluate()` untuk cetak ROC-AUC, PR-AUC, classification report |
| 7 | Logistic Regression | Training baseline + logging ke MLflow |
| 8 | XGBoost + Optuna | 20 trials Optuna tuning + MLflow tracking per trial |
| 9 | Final XGBoost | Train model final + feature importance plot |
| 10 | Perbandingan & Simpan | Tabel & bar chart semua model, simpan model ke Drive |
| 11 | Test Inference | Generate prediksi fraud jika `test_transaction.csv` tersedia; jika tidak, tampilkan validation summary + simpan `val_predictions.csv` |

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
optuna
mlflow
joblib
```
