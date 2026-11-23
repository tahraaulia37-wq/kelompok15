# 🌙 Insomnia Risk Prediction System (Kelompok 15)

Aplikasi web berbasis *Machine Learning* untuk mendeteksi tingkat risiko insomnia (**Low**, **Moderate**, dan **High Risk**) berdasarkan pola tidur, penggunaan gawai, serta beban stres-kelelahan harian.

🔗 **Live Demo Aplikasi:** [kelompok15-insomnia-risk.streamlit.app](https://kelompok15-insomnia-risk.streamlit.app)

---

## 👥 Anggota Kelompok 15
* **Tahra Aulia** – J0404241063
* **Fakhriyatullahi Zahwa Khaerunnisa	– J0404241147
* **Muhammad Bahrain Sekai Suryawi** – J0404241171

---

## ⚙️ Ringkasan Teknis Model
* **Algoritma Utama:** XGBoost Classifier (`XGBClassifier`)
* **Fitur Input:** 10 parameter gaya hidup & tidur + 2 fitur hasil *Feature Engineering*:
  1. `bedtime_screen_ratio`: Rasio durasi penggunaan layar sebelum tidur terhadap total waktu layar harian.
  2. `combined_stress_fatigue`: Indeks interaksi gabungan antara tingkat stres dan kelelahan harian.
* **Penanganan Data & Kalibrasi:**
  * Penyeimbangan kelas menggunakan **SMOTE** dan pembobotan sampel (*sample weights*).
  * **Threshold Tuning** (`High Risk ≥ 0.40`, `Moderate Risk ≥ 0.31`) untuk memaksimalkan sensitivitas (*recall*) deteksi dini risiko insomnia.

---

## 📂 Struktur Proyek
```text
kelompok15/
└── insomnia-risk/
    ├── app.py                     # Kode utama antarmuka web Streamlit
    ├── model_insomnia_xgb.pkl     # Model XGBoost terlatih
    ├── preprocessor.pkl           # Pipeline preprocessing & scaling fitur
    ├── label_encoder.pkl          # Encoder label kelas risiko
    └── requirements.txt           # Daftar dependensi Python
