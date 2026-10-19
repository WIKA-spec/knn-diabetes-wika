import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="SPK Risiko Diabetes | KNN", page_icon="🩺", layout="wide")

# Dataset sesuai Studi_Kasus_KNN.pdf
DATA = {
    "Pasien": ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"],
    "Usia": [25, 30, 35, 40, 45, 50, 55, 28, 38, 48],
    "BMI": [21.5, 23.1, 25.0, 27.5, 30.2, 31.5, 33.0, 22.5, 26.8, 29.5],
    "Glukosa": [90, 95, 105, 120, 140, 155, 165, 92, 115, 145],
    "Tekanan_Darah": [110, 115, 120, 125, 135, 140, 145, 112, 122, 130],
    "Riwayat_Keluarga": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    "Kelas": ["Rendah", "Rendah", "Rendah", "Tinggi", "Tinggi", "Tinggi", "Tinggi", "Rendah", "Rendah", "Tinggi"]
}
df = pd.DataFrame(DATA)
FITUR = ["Usia", "BMI", "Glukosa", "Tekanan_Darah", "Riwayat_Keluarga"]

st.title("🩺 Sistem Pendukung Keputusan Risiko Diabetes")
st.subheader("Metode K-Nearest Neighbors (KNN)")
st.write("Aplikasi demonstrasi Data Mining untuk mengklasifikasikan data pasien ke kelas **Risiko Rendah** atau **Risiko Tinggi**.")

st.warning("Aplikasi ini hanya untuk pembelajaran, bukan alat diagnosis medis. Dataset hanya berisi 10 contoh pasien dan tidak cukup untuk keputusan klinis nyata.")

with st.sidebar:
    st.header("Input Data Pasien")
    usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=42, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=28.5, step=0.1, format="%.1f")
    glukosa = st.number_input("Glukosa", min_value=20, max_value=600, value=130, step=1)
    tekanan = st.number_input("Tekanan darah", min_value=30, max_value=300, value=128, step=1)
    riwayat = st.selectbox("Riwayat diabetes dalam keluarga", [0, 1],
                          format_func=lambda x: "Ada (1)" if x == 1 else "Tidak ada (0)")
    k = st.slider("Jumlah tetangga (K)", min_value=1, max_value=9, value=3, step=2)
    jalankan = st.button("Analisis Risiko", type="primary", use_container_width=True)

tab_data, tab_hasil, tab_info = st.tabs(["📋 Data Latih", "🔎 Analisis Pasien", "📘 Penjelasan Metode"])

with tab_data:
    st.markdown("### Dataset 10 pasien dari soal")
    st.dataframe(df, use_container_width=True, hide_index=True)
    a, b = st.columns(2)
    a.metric("Jumlah data latih", len(df))
    b.metric("Jumlah atribut", len(FITUR))
    st.caption("Riwayat keluarga: 0 = tidak ada, 1 = ada. Kelas: Rendah atau Tinggi.")

pasien_baru = pd.DataFrame([{
    "Usia": usia, "BMI": bmi, "Glukosa": glukosa,
    "Tekanan_Darah": tekanan, "Riwayat_Keluarga": riwayat
}])

# Standardisasi menghindari atribut berskala besar mendominasi jarak.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=k, metric="euclidean"))
])
model.fit(df[FITUR].astype(float), df["Kelas"])
prediksi = model.predict(pasien_baru[FITUR])[0]
kelas_model = list(model.named_steps["knn"].classes_)
prob = model.predict_proba(pasien_baru[FITUR])[0]
proporsi_tetangga = float(prob[kelas_model.index(prediksi)]) * 100

scaler = model.named_steps["scaler"]
x_latih = scaler.transform(df[FITUR].astype(float))
x_baru = scaler.transform(pasien_baru[FITUR])
jarak = np.sqrt(((x_latih - x_baru) ** 2).sum(axis=1))
hasil_jarak = df.copy()
hasil_jarak["Jarak standar"] = jarak
hasil_jarak = hasil_jarak.sort_values("Jarak standar").reset_index(drop=True)
tetangga = hasil_jarak.head(k)

with tab_hasil:
    st.markdown("### Data pasien yang dianalisis")
    st.dataframe(pasien_baru, use_container_width=True, hide_index=True)
    if jalankan:
        st.markdown("### Hasil klasifikasi")
        if prediksi == "Tinggi":
            st.error("Prediksi KNN: RISIKO TINGGI")
        else:
            st.success("Prediksi KNN: RISIKO RENDAH")
        st.caption(f"Proporsi tetangga yang mendukung kelas prediksi: {proporsi_tetangga:.1f}%. Ini bukan probabilitas klinis.")
        st.markdown(f"### {k} tetangga terdekat")
        st.dataframe(
            tetangga[["Pasien", "Usia", "BMI", "Glukosa", "Tekanan_Darah", "Riwayat_Keluarga", "Jarak standar", "Kelas"]]
            .style.format({"BMI": "{:.1f}", "Jarak standar": "{:.4f}"}),
            use_container_width=True, hide_index=True
        )
        suara = tetangga["Kelas"].value_counts().reindex(["Rendah", "Tinggi"], fill_value=0)
        st.markdown("### Jumlah suara setiap kelas")
        st.bar_chart(suara)
        st.markdown("### Jarak seluruh pasien latih")
        st.bar_chart(hasil_jarak.set_index("Pasien")["Jarak standar"])
    else:
        st.info("Atur data pasien pada panel kiri, lalu klik **Analisis Risiko**.")

with tab_info:
    st.markdown("### Tahapan algoritma KNN")
    st.markdown("""
    1. Menggunakan 10 pasien berlabel sebagai data latih.
    2. Menstandardisasi atribut karena skala usia, BMI, glukosa, tekanan darah, dan riwayat keluarga berbeda.
    3. Menghitung jarak Euclidean pasien baru ke setiap pasien latih.
    4. Memilih K pasien dengan jarak paling kecil.
    5. Mengambil kelas mayoritas sebagai hasil klasifikasi.
    """)
    st.latex(r"d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}")
    st.info("PDF tugas tidak menyebutkan normalisasi. Aplikasi ini menggunakan StandardScaler agar atribut berskala besar tidak mendominasi jarak, sehingga hasil bisa berbeda dari perhitungan jarak mentah.")

st.divider()
st.caption("Proyek tugas Data Mining • Python + Streamlit • Bukan alat diagnosis medis")
