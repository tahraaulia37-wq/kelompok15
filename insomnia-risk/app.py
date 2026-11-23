import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Insomnia Risk AI | Analisis Kesehatan Tidur",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik kartu metrik dan tombol
st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 15px;
        border-radius: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load Model & Assets
@st.cache_resource
def load_assets():
    model = joblib.load('model_insomnia_xgb.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, preprocessor, label_encoder

model, preprocessor, label_encoder = load_assets()

# 3. Sidebar Informasi & Panduan Pengujian
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3094/3094837.png", width=80)
    st.title("Insomnia Risk")
    st.divider()
    
    st.subheader("📌 Tentang Model")
    st.info(
        "Model ini menganalisis **10 parameter gaya hidup** serta **2 fitur interaksi** "
        "(Rasio Penggunaan Gawai Malam & Beban Stres-Mental) untuk mendeteksi potensi insomnia lebih dini."
    )
    
    st.subheader("🧪 Panduan Skala Input")
    st.markdown("""
    * **1 – 3:** Rendah / Sangat Baik
    * **4 – 7:** Sedang / Wajar
    * **8 – 10:** Tinggi / Berat
    """)

# 4. Header Utama
st.title("🌙 Analisis & Prediksi Risiko Insomnia")
st.markdown("Lengkapi parameter kebiasaan harian dan paparan layar Anda di bawah ini untuk mendapatkan evaluasi tingkat risiko tidur.")
st.divider()

# 5. Layout Dua Kolom: Kiri (Input Form), Kanan (Hasil Analisis)
col_input, col_result = st.columns([1.15, 1], gap="large")

with col_input:
    st.subheader("📝 Parameter Kebiasaan Harian")
    
    with st.form("form_prediksi"):
        tab1, tab2 = st.tabs(["👤 Profil & Gaya Hidup", "📱 Paparan Layar & Stres"])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Usia (Tahun)", min_value=10, max_value=100, value=25)
                gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])
                occupation = st.selectbox(
                    "Pekerjaan", 
                    ["Student", "Engineer", "Doctor", "Teacher", "Artist", "Manager", "Other"]
                )
            with c2:
                caffeine_cups = st.number_input("Konsumsi Kafein (Cangkir/Hari)", min_value=0, max_value=15, value=1)
                physical_activity = st.number_input("Aktivitas Fisik (Menit/Hari)", min_value=0, max_value=300, value=30)

        with tab2:
            c3, c4 = st.columns(2)
            with c3:
                daily_screen_time = st.number_input("Total Screen Time Harian (Jam)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
                phone_before_sleep = st.number_input("Main HP Sebelum Tidur (Menit)", min_value=0, max_value=300, value=45)
                notifications = st.number_input("Notifikasi Diterima / Hari", min_value=0, max_value=500, value=60)
            with c4:
               # 1. Slider Tingkat Stres + Keterangan Dinamis
                stress_level = st.slider(
                    "Tingkat Stres Harian (1 - 10)", 
                    min_value=1, max_value=10, value=5,
                    help="1-2: Sangat tenang | 3-4: Sedikit tegang | 5-6: Tertekan wajar | 7-8: Cemas/tegang berat | 9-10: Sangat tertekan/panik"
                )
                ket_stres = {
                    1: "😌 Sangat rileks, pikiran tenang tanpa beban.",
                    2: "🙂 Tenang, aktivitas berjalan lancar.",
                    3: "😐 Sedikit tekanan ringan, masih mudah dikendalikan.",
                    4: "🤔 Mulai terasa beban tugas/pekerjaan.",
                    5: "😕 Stres sedang, sesekali merasa cemas/terburu-buru.",
                    6: "😣 Cukup tertekan, pikiran mulai sulit santai.",
                    7: "😫 Stres tinggi, mudah gelisah dan sulit rileks.",
                    8: "😰 Sangat tegang, beban pikiran terasa menumpuk.",
                    9: "🤯 Hampir kewalahan, cemas terus-menerus.",
                    10: "🆘 Stres ekstrem / panik berat sepanjang hari."
                }
                st.caption(f"**Arti skala {stress_level}:** {ket_stres[stress_level]}")

                st.markdown("<br>", unsafe_allow_html=True)

                # 2. Slider Kelelahan Mental + Keterangan Dinamis
                mental_fatigue = st.slider(
                    "Skor Kelelahan Mental (1 - 10)", 
                    min_value=1, max_value=10, value=5,
                    help="1-2: Fokus tajam & segar | 3-4: Sedikit lelah | 5-6: Konsentrasi mulai turun | 7-8: Sulit fokus/jenuh | 9-10: Burnout total"
                )
                ket_fatigue = {
                    1: "⚡ Pikiran sangat segar, fokus & konsentrasi penuh.",
                    2: "🔋 Energi mental baik, siap berpikir berat.",
                    3: "🙂 Sedikit lelah setelah beraktivitas, cepat pulih.",
                    4: "😐 Mulai terasa jenuh ringan di sore hari.",
                    5: "🥱 Lelah mental sedang, butuh jeda istirahat.",
                    6: "😶‍🌫️ Konsentrasi mulai menurun, agak lambat berpikir.",
                    7: "😵‍💫 Lelah secara emosional/kognitif, sulit fokus lama.",
                    8: "🪫 Sangat jenuh, mudah lupa dan kehilangan motivasi.",
                    9: "🥀 Mendekati burnout, kepala terasa penuh/berat.",
                    10: "🛑 Burnout total, otak terasa buntu & kelelahan ekstrem."
                }
                st.caption(f"**Arti skala {mental_fatigue}:** {ket_fatigue[mental_fatigue]}")

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🔍 Analisis Risiko Insomnia Sekarang", type="primary")

with col_result:
    st.subheader("📊 Hasil Evaluasi & Rekomendasi")
    
    if not submit:
        st.info("👈 Silakan isi data di sebelah kiri dan klik **Analisis Risiko Insomnia Sekarang** untuk melihat hasil diagnosa.")
    else:
        # Susun DataFrame
        input_data = pd.DataFrame([{
            'age': age,
            'gender': gender,
            'occupation': occupation,
            'daily_screen_time_hours': daily_screen_time,
            'phone_usage_before_sleep_minutes': phone_before_sleep,
            'stress_level': stress_level,
            'caffeine_intake_cups': caffeine_cups,
            'physical_activity_minutes': physical_activity,
            'notifications_received_per_day': notifications,
            'mental_fatigue_score': mental_fatigue
        }])

        # Hitung 2 fitur interaksi
        bedtime_ratio = phone_before_sleep / (daily_screen_time * 60 + 1)
        stress_fatigue = stress_level * mental_fatigue
        
        input_data['bedtime_screen_ratio'] = bedtime_ratio
        input_data['combined_stress_fatigue'] = stress_fatigue

        # Prediksi Probabilitas
        X_processed = preprocessor.transform(input_data)
        pred_proba = model.predict_proba(X_processed)[0]

        # Threshold Tuning
        classes = list(label_encoder.classes_)
        idx_high = classes.index('High')
        idx_mod = classes.index('Moderate')
        idx_low = classes.index('Low')

        # Prioritaskan High hanya jika probabilitasnya >= 0.40 DAN lebih besar dari Low
        if pred_proba[idx_high] >= 0.40 and pred_proba[idx_high] > pred_proba[idx_low]:
            pred_label = 'High'
        # Prioritaskan Moderate jika probabilitasnya >= 0.30 dan High tidak mendominasi
        elif pred_proba[idx_mod] >= 0.31 and pred_proba[idx_low] < 0.55:
            pred_label = 'Moderate'
        else:
            pred_label = label_encoder.inverse_transform([np.argmax(pred_proba)])[0]

        # Banner Status Utama
        if pred_label == 'High':
            st.error(f"### 🚨 TINGKAT RISIKO: TINGGI (HIGH RISK)\nTerdeteksi indikasi kuat gangguan pola tidur dengan probabilitas risiko tinggi sebesar **{pred_proba[idx_high]*100:.1f}%**.")
        elif pred_label == 'Moderate':
            st.warning(f"### ⚠️ TINGKAT RISIKO: SEDANG (MODERATE RISK)\nPola kebiasaan Anda mulai membebani kualitas istirahat (Probabilitas Moderate: **{pred_proba[idx_mod]*100:.1f}%**).")
        else:
            st.success(f"### ✅ TINGKAT RISIKO: RENDAH (LOW RISK)\nKebiasaan harian dan manajemen stres Anda berada dalam kategori sehat untuk kualitas tidur optimal.")

        # Kartu Metrik Fitur Rekayasa
        m1, m2 = st.columns(2)
        m1.metric(
            label="Rasio Layar Sebelum Tidur", 
            value=f"{bedtime_ratio*100:.1f}%",
            help="Proporsi waktu menatap layar menjelang tidur dibandingkan total screen time harian."
        )
        m2.metric(
            label="Indeks Beban Stres-Mental", 
            value=f"{stress_fatigue} / 100",
            help="Hasil perkalian Tingkat Stres dan Kelelahan Mental (Ideal < 25)."
        )

        # Grafik Distribusi Probabilitas
        st.markdown("**Distribusi Probabilitas Model:**")
        chart_df = pd.DataFrame({
            'Kategori Risiko': classes,
            'Probabilitas (%)': [round(p * 100, 2) for p in pred_proba]
        }).set_index('Kategori Risiko')
        st.bar_chart(chart_df, height=200)

        # Rekomendasi Personal Otomatis
        with st.expander("💡 Lihat Rekomendasi Tindakan Personal", expanded=True):
            rekomendasi = []
            if phone_before_sleep > 30:
                rekomendasi.append("📵 **Digital Detox Malam:** Kurangi penggunaan HP minimal 30–60 menit sebelum jam tidur untuk mencegah supresi hormon melatonin akibat cahaya biru.")
            if stress_fatigue >= 36:
                rekomendasi.append("🧘 **Manajemen Stres & Relaksasi:** Skor beban mental-stres Anda cukup tinggi. Lakukan teknik pernapasan dalam (4-7-8) atau peregangan ringan sebelum tidur.")
            if caffeine_cups >= 3:
                rekomendasi.append("☕ **Batasi Kafein:** Hindari konsumsi kopi/teh berkafein setelah pukul 14.00 siang.")
            if physical_activity < 20:
                rekomendasi.append("🏃 **Tingkatkan Aktivitas Fisik:** Tambahkan jalan cepat atau olahraga ringan minimal 20–30 menit di pagi/sore hari untuk membantu regulasi ritme sirkadian.")
            
            if not rekomendasi:
                st.write("Pertahankan rutinitas tidur teratur dan pola hidup sehat Anda saat ini!")
            else:
                for item in rekomendasi:
                    st.markdown(f"- {item}")