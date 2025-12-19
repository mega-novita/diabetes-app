# ==============================================================================
# 1. IMPORT LIBRARY
# ==============================================================================
# Streamlit: Framework untuk membuat web app data science dengan mudah
import streamlit as st
# Pandas: Untuk mengelola data dalam bentuk tabel (DataFrame)
import pandas as pd
# Numpy: Untuk operasi matematika angka
import numpy as np
# Pickle: Untuk memuat file model yang sudah disimpan (.pkl)
import pickle
# Time: Untuk memberikan efek jeda/loading
import time
# Matplotlib & Seaborn: Untuk membuat grafik visualisasi
import matplotlib.pyplot as plt
import seaborn as sns
# Sklearn: Diperlukan di sini hanya untuk demonstrasi preprocessing di halaman EDA
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==============================================================================
# 2. KONFIGURASI HALAMAN
# ==============================================================================
# Mengatur judul tab browser, icon, dan layout halaman agar lebar (wide)
st.set_page_config(
    layout="wide", 
    page_title="Aplikasi Prediksi Diabetes punya Mega", 
    page_icon="🩺"
)

# ==============================================================================
# 3. FUNGSI LOAD DATA & MODEL (CACHING)
# ==============================================================================
# @st.cache_data: Agar data hanya dibaca sekali saja saat awal, biar aplikasi cepat.
# Tidak perlu baca ulang setiap kali kita klik tombol.
@st.cache_data
def load_data():
    try:
        # Membaca file CSV
        return pd.read_csv('diabetes_dataset.csv')
    except:
        return None

# @st.cache_resource: Sama seperti cache_data, tapi khusus untuk objek berat seperti Model AI.
@st.cache_resource
def load_model_artifact():
    try:
        # Membuka file 'best_diabetes_model_tuned.pkl' (sesuaikan nama file model Anda)
        with open('diabetes_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

# Memanggil fungsi di atas untuk mendapatkan data & model
df = load_data()
artifact = load_model_artifact()

# ==============================================================================
# 4. SIDEBAR NAVIGATION (MENU SAMPING)
# ==============================================================================
st.sidebar.title("Navigasi")
# Membuat menu pilihan halaman
nav = st.sidebar.selectbox(
    "Pilih Menu", 
    ("Home", "Dataset", "Exploratory Data Analysis", "Modelling", "Prediction", "About")
)

# ==============================================================================
# 5. HALAMAN 1: HOME
# ==============================================================================
if nav == "Home":
    st.title("🏥 Klasifikasi Risiko Diabetes")
    st.write("""
    **Selamat Datang!**
    
    Aplikasi ini dibuat untuk membantu orang awam mengecek risiko diabetes secara dini
    menggunakan kecerdasan buatan (*Artificial Intelligence*).
    
    Kami menggunakan data medis standar seperti Gula Darah, BMI, dan HbA1c untuk
    melakukan prediksi.
    """)
    
    # Menampilkan gambar ilustrasi dari internet
    
    st.info("💡 **Panduan Menu:**")
    st.info("Pilih menu 'Dataset' untuk melihat sample dataset yang digunakan dalam pemodelan.")
    st.info("Pilih menu 'Exploratory Data Analysis' untuk melihat Visualisasi Data.")
    st.info("Pilih menu 'Modelling' untuk melihat hasil perbandingan model yang digunakan.")
    st.info("Pilih menu 'Prediction' untuk memulai diagnosa.")
    st.info("Pilih menu 'About' untuk melihat profil.")

# ==============================================================================
# 6. HALAMAN 2: DATASET
# ==============================================================================
elif nav == "Dataset":
    st.header("📂 Tinjauan Dataset")
    
    if df is not None:
        st.write("""
        **1. Tentang Apa Dataset Ini?**
                 
        Dataset ini berisi data rekam medis pasien yang bertujuan untuk mendiagnosis atau memprediksi risiko penyakit Diabetes (Tipe 2). 
        
        Data ini menghubungkan kebiasaan hidup seseorang (makan, tidur, olahraga) 
        dengan hasil laboratorium medis mereka (gula darah, kolesterol) 
        untuk menentukan apakah mereka menderita diabetes atau tidak.
        """)
        st.write("""
        **2. Target Variable (Label/Kunci Jawaban)**
        Ini adalah kolom yang biasanya ingin kita prediksi menggunakan Machine Learning:

        diagnosed_diabetes (Target Utama):

        Tipe: Kategorikal / Biner (1 atau 0).

        Isi: 1 (Positif Diabetes) atau 0 (Negatif/Sehat).

        Kegunaan: Untuk Supervised Learning jenis Klasifikasi.

        Target Alternatif (Opsional):

        diabetes_stage: Jika ingin memprediksi tahapan (misal: "No Diabetes", "Pre-diabetes", "Type 2"). Ini untuk Multi-class Classification.

        diabetes_risk_score: Jika ingin memprediksi skor risiko (angka kontinu). Ini untuk Regression.
        """)

        st.write("""
        **3. Fitur (Features / Input Variables)**
        Kolom-kolom ini adalah "Soal" atau data pendukung untuk melakukan prediksi. Kita bisa mengelompokkannya menjadi 3 kategori besar:

        A. Data Demografis & Sosial (Siapa Pasiennya?)
        age: Umur pasien.

        gender: Jenis kelamin (Male/Female).

        ethnicity: Suku bangsa/Etnis.

        education_level: Tingkat pendidikan.

        income_level: Tingkat pendapatan.

        employment_status: Status pekerjaan.

        B. Data Gaya Hidup (Lifestyle) - Faktor Risiko Eksternal
        smoking_status: Status merokok (Never, Former, Current).

        alcohol_consumption_per_week: Konsumsi alkohol mingguan.

        physical_activity_minutes_per_week: Durasi olahraga per minggu.

        diet_score: Skor kualitas diet/makan (kemungkinan skala angka).

        sleep_hours_per_day: Jam tidur per hari.

        screen_time_hours_per_day: Lama menatap layar (HP/Laptop) per hari.

        C. Data Riwayat Medis & Klinis (Hasil Lab) - Faktor Risiko Internal
        Riwayat Penyakit:

        family_history_diabetes: Apakah ada keluarga yang diabetes? (0/1).

        hypertension_history: Riwayat darah tinggi.

        cardiovascular_history: Riwayat penyakit jantung.

        Pengukuran Fisik:

        bmi (Body Mass Index): Indeks massa tubuh (Berat/Tinggi). Indikator obesitas.

        waist_to_hip_ratio: Rasio lingkar pinggang dan pinggul (indikator lemak perut).

        systolic_bp & diastolic_bp: Tekanan darah (atas dan bawah).

        heart_rate: Detak jantung.

        Laboratorium Darah (PENTING):

        glucose_fasting: Gula darah puasa.

        glucose_postprandial: Gula darah setelah makan.

        hba1c: Rata-rata gula darah dalam 3 bulan terakhir (Indikator emas diabetes).

        insulin_level: Kadar insulin.

        cholesterol_total, hdl, ldl, triglycerides: Profil lemak darah.
        """)

        st.write(f"**Total Dataset**")
        st.write(f"**Jumlah Baris Data:** {df.shape[0]}")
        st.write(f"**Jumlah Kolom Fitur:** {df.shape[1]}")

        st.write("Ini adalah contoh data mentah yang digunakan untuk melatih model AI:")
        # Menampilkan 10 baris pertama data
        st.dataframe(df.head(10))
        
        # # Membuat 2 kolom berdampingan untuk info statistik
        # col1, col2 = st.columns(2)
        # with col1:
        # with col2:
        st.write("**Statistik Singkat:**")
        st.dataframe(df.describe())
    else:
        st.error("File 'diabetes_dataset.csv' tidak ditemukan. Harap upload file tersebut.")

# ==============================================================================
# 7. HALAMAN 3: EDA (EXPLORATORY DATA ANALYSIS)
# ==============================================================================
elif nav == "Exploratory Data Analysis":
    st.header("📊 Analisis Data Eksploratif & Preprocessing")
    
    if df is not None:
        # --- TAB 1: VISUALISASI ---
        st.subheader("1. Visualisasi Data")
        tab1, tab2 = st.tabs(["Distribusi Target", "Hubungan Fitur"])
        
        with tab1:
            st.write("**Perbandingan Orang Sehat vs Diabetes**")
            fig1, ax1 = plt.subplots(figsize=(6,4))
            # Menggambar diagram batang jumlah data
            sns.countplot(x='diagnosed_diabetes', data=df, palette='pastel', ax=ax1)
            ax1.set_xticklabels(['Sehat (0)', 'Diabetes (1)'])
            st.pyplot(fig1) # Menampilkan plot ke Streamlit
            
        with tab2:
            st.write("**Hubungan Gula Darah dengan Umur**")
            fig2, ax2 = plt.subplots(figsize=(8,4))
            # Scatterplot untuk melihat sebaran titik data
            sns.scatterplot(x='age', y='glucose_fasting', hue='diagnosed_diabetes', data=df, palette='coolwarm', ax=ax2)
            st.pyplot(fig2)

        st.divider() # Garis pembatas

        # --- TAB 2: PREPROCESSING (PERMINTAAN NO 1) ---
        st.subheader("2. Proses Pengolahan Data (Preprocessing)")
        st.write("""
        Sebelum data masuk ke mesin AI, data harus 'dimasak' dulu agar mesin mengerti.
        Proses ini disebut **Preprocessing**.
        """)

        # Tahap A: Data Mentah
        st.write("#### Langkah A: Data Awal (Masih Bahasa Manusia)")
        st.dataframe(df[['gender', 'age', 'glucose_fasting']].head(3))
        st.caption("Lihat kolom 'gender', isinya masih teks 'Male'/'Female'. Komputer tidak bisa hitung teks.")

        # Tahap B: Encoding & Scaling (Simulasi)
        # Kita buat copy data biar data asli tidak rusak
        df_demo = df[['gender', 'age', 'glucose_fasting']].head(5).copy()
        
        # Encoding: Ubah Teks -> Angka
        le_demo = LabelEncoder()
        df_demo['gender_encoded'] = le_demo.fit_transform(df_demo['gender'])
        
        # Scaling: Ubah Angka -> Skala Standar
        scaler_demo = StandardScaler()
        # Kita scale umur dan gula darah
        scaled_values = scaler_demo.fit_transform(df_demo[['age', 'glucose_fasting']])
        df_demo['age_scaled'] = scaled_values[:, 0]
        df_demo['glucose_scaled'] = scaled_values[:, 1]

        st.write("#### Langkah B: Data Setelah Preprocessing (Bahasa Mesin)")
        st.dataframe(df_demo[['gender_encoded', 'age_scaled', 'glucose_scaled']].head(3))
        st.caption("""
        **Perubahan:**
        1. **Encoding:** 'Male/Female' berubah jadi angka 0 atau 1.
        2. **Scaling:** Umur '50' dan Gula '150' diubah jadi angka desimal kecil (misal -0.5 sampai 1.2) agar setara.
        """)

    else:
        st.warning("Data tidak tersedia.")

# ==============================================================================
# 8. HALAMAN 4: MODELLING
# ==============================================================================
elif nav == "Modelling":
    st.header("⚙️ Performa Model AI")
    
    st.write("""
    Kami melatih 3 algoritma berbeda: **KNN**, **Naive Bayes**, dan **Decision Tree**.
    Proses pelatihan dilakukan dalam 2 tahap:
    1. **Baseline:** Menggunakan pengaturan standar (pabrikan).
    2. **Tuned:** Menggunakan pengaturan yang sudah dioptimalkan (hyperparameter tuning).
    """)
    
    if artifact and 'history' in artifact:
        # 1. Mengambil data history dari file pickle
        history_data = artifact['history']
        
        # 2. Membuat DataFrame
        df_compare = pd.DataFrame(history_data)
        
        # 3. Formatting Tampilan Angka (Agar jadi persen %)
        # Kita format kolom angka agar terlihat cantik (misal 0.82 jadi 82.00%)
        st.subheader("Tabel Perbandingan Akurasi")
        
        # Menggunakan Styler untuk format
        st.dataframe(
            df_compare.style.format({
                'Akurasi Awal': '{:.2%}',
                'Akurasi Tuned': '{:.2%}',
                'Improvement': '{:+.2%}' # Tanda + biar kelihatan naik/turun
            })
        )
        
        # 4. Membuat Grafik Perbandingan
        st.subheader("Grafik Peningkatan Performa")
        
        # Kita reshape data biar mudah di-plot (Melt)
        df_melted = df_compare.melt(id_vars="Model", value_vars=["Akurasi Awal", "Akurasi Tuned"], var_name="Kondisi", value_name="Akurasi")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_melted, x="Model", y="Akurasi", hue="Kondisi", palette="viridis", ax=ax)
        ax.set_ylim(0, 1.0) # Set batas Y dari 0 sampai 100%
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        st.pyplot(fig)
        
        st.success(f"🏆 **Model Terpilih:** {artifact['model_name']} dengan akurasi **{artifact['accuracy']:.2%}**")
        
    else:
        st.warning("⚠️ Data histori perbandingan tidak ditemukan di dalam file model. Harap jalankan ulang `models.py` yang baru.")
    

# ==============================================================================
# 9. HALAMAN 5: PREDICTION
# ==============================================================================
elif nav == "Prediction":
    st.header("🔮 Prediksi Risiko Diabetes")
    st.markdown("Silakan isi formulir di bawah ini sesuai data pasien.")
    
    if artifact:
        model = artifact['model']
        scaler = artifact['scaler']
        # Panggil kedua encoder
        enc_gender = artifact['encoder_gender']
        enc_smoking = artifact['encoder_smoking'] # Ambil encoder rokok
        
        with st.form("pred_form"):
            st.subheader("Data Pasien")
            c1, c2, c3 = st.columns(3) # Bagi jadi 3 kolom biar rapi
            
            # --- KOLOM 1: DEMOGRAFI & FISIK ---
            with c1:
                st.markdown("##### 1. Fisik & Demografi")
                age = st.number_input("Umur", 20, 90, 45)
                gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
                # Input BMI otomatis
                weight = st.number_input("Berat (kg)", 30, 150, 70)
                height = st.number_input("Tinggi (cm)", 100, 250, 170)
                bmi = weight / ((height/100)**2)
                st.caption(f"BMI Terhitung: {bmi:.1f}")

            # --- KOLOM 2: GAYA HIDUP (FITUR BARU) ---
            with c2:
                st.markdown("##### 2. Gaya Hidup")
                # Mengambil opsi rokok langsung dari encoder biar cocok 100%
                smoking_options = ['Perokok', 'Sudah Berhenti Merokok', 'Tidak Pernah Merokok'] 
                smoking = st.selectbox("Status Merokok", smoking_options)

                if (smoking == 'Perokok'):
                    smoking = 'Current'
                elif (smoking == 'Sudah Berhenti Merokok'):
                    smoking = 'Former'
                else :
                    smoking = 'Never'
                
                activity = st.number_input("Olahraga (menit/minggu)", 0, 1000, 150, help="Saran WHO: Minimal 150 menit/minggu")
                
                # Riwayat Keluarga & Hipertensi
                family = st.selectbox("Riwayat Diabetes Keluarga?", ["No", "Yes"])
                hyper = st.selectbox("Riwayat Hipertensi?", ["No", "Yes"])

            # --- KOLOM 3: MEDIS & LAB (FITUR BARU) ---
            with c3:
                st.markdown("##### 3. Cek Laboratorium")
                glucose = st.number_input("Gula Darah Puasa (mg/dL)", 50, 400, 100)
                # hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5)
                cholesterol = st.number_input("Kolesterol Total (mg/dL)", 100, 400, 180, help="Normal: < 200 mg/dL")
            
            # Tombol untuk memulai prediksi
            submit_btn = st.form_submit_button("🔍 Analisis Risiko Sekarang")

        # --- LOGIKA KETIKA TOMBOL DITEKAN ---
        if submit_btn:
            # 1. Hitung BMI (Rumus: Berat / Tinggi Meter Kuadrat)
            height_m = height / 100
            bmi_calculated = weight / (height_m ** 2)
            st.info(f"ℹ️ BMI Pasien Terhitung: **{bmi_calculated:.2f}**")
            
            # 2. Terjemahkan Input User (Preprocessing)
            # Ubah Male/Female jadi angka
            gender_val = enc_gender.transform([gender])[0]
            smoke_val = enc_smoking.transform([smoking])[0]
            # Ubah Yes/No jadi 1/0
            family_val = 1 if family == "Yes" else 0
            hyper_val = 1 if hyper == "Yes" else 0
            
            # 3. Susun Data Input sesuai urutan saat Training
            # Urutan: [age, gender, bmi, glucose, hba1c, family, hyper]
            # input_data = [[
            #     age, gender_val, bmi, glucose, hba1c, family_val, hyper_val, 
            #     smoke_val, activity, cholesterol
            # ]]
            input_data = [[
                age, gender_val, bmi, glucose, family_val, hyper_val, 
                smoke_val, activity, cholesterol
            ]]
            
            # 4. Scaling (PENTING!)
            # Data baru harus disetarakan skalanya menggunakan scaler yang sama dengan training
            input_scaled = scaler.transform(input_data)
            
            # 5. Prediksi oleh Model
            with st.spinner('Sedang menganalisis pola kesehatan...'):
                time.sleep(1) # Efek loading buatan
                prediction = model.predict(input_scaled)[0]     # Hasil 0 atau 1
                probs = model.predict_proba(input_scaled)       # Persentase keyakinan
            
            # 6. Tampilkan Hasil
            st.divider()
            confidence = probs[0][prediction] * 100
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                # Tampilkan ikon sesuai hasil
                if prediction == 1:
                    st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=120)
                else:
                    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966334.png", width=120)

            with col_res2:
                if prediction == 1:
                    st.error(f"### HASIL: POSITIF (Berisiko)")
                    st.write("Sistem mendeteksi adanya pola yang mirip dengan pasien diabetes.")
                    st.markdown("**Saran:** Segera konsultasi ke dokter dan atur pola makan.")
                else:
                    st.success(f"### HASIL: NEGATIF (Sehat)")
                    st.write("Sistem tidak mendeteksi risiko diabetes yang signifikan.")
                    st.markdown("**Saran:** Pertahankan gaya hidup sehat dan olahraga teratur.")

# ==============================================================================
# 10. HALAMAN 6: ABOUT
# ==============================================================================
elif nav == "About":
    st.header("Tentang Saya")
    st.image("photos/foto-profil.jpg", width=150)
    
    st.write("""
    **Assoc. Prof. Mega Novita**
    
    Saya adalah seorang akademisi dan peneliti terkemuka yang saat ini menjabat sebagai Kepala Kantor Urusan Internasional (Head of International Affairs) di Universitas PGRI Semarang (UPGRIS). 
             
    Dengan memiliki latar belakang pendidikan yang kuat dengan gelar MSc dan PhD di bidang Kimia dari Kwansei Gakuin University, Jepang, serta pengalaman postdoctoral di Jepang dan Korea Selatan.
    Menguasai keahlian secara interdisipliner, mencakup bidang kimia, biologi, dan matematika.

    Dalam dunia pendidikan nasional, saya berperan aktif dalam reformasi pendidikan di Indonesia. 
    Sejak tahun 2022, telah menjadi fasilitator bagi lebih dari 60 sekolah dalam Program Sekolah Penggerak dan membimbing lebih dari 150 mahasiswa sebagai Dosen Pembimbing Lapangan di Program Kampus Mengajar.

    Di ranah ilmiah, Mega Novita aktif dalam kolaborasi riset internasional.
    Fokus penelitiannya mendalami terbagi menjadi 2 yaitu pada bidang Informatika dan Kimia.
             
    Pada Bidang Informatika berupa penggunaan data science sederhana.
    Dan pada bidang Kimia mengenai struktur elektronik logam transisi dan ion tanah jarang (rare-earth ions), khususnya untuk material fosfor merah.
    """)
    st.markdown("---")

    st.caption("© 2025 Diabetes Prediction Project")

