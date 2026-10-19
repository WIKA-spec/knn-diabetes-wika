"""
Sistem Pendukung Keputusan - Prediksi Risiko Diabetes
Algoritma : K-Nearest Neighbors (KNN) - diimplementasikan manual (tanpa scikit-learn)
Jalankan  : streamlit run app.py
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SPK Risiko Diabetes (KNN)", page_icon="🩺", layout="wide")

FITUR = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]
LABEL = {0: "Rendah", 1: "Tinggi"}

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
DATA_DEFAULT = pd.DataFrame(
    {
        "Pasien": ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"],
        "Usia": [25, 30, 35, 40, 45, 50, 55, 28, 38, 48],
        "BMI": [21.5, 23.1, 25.0, 27.5, 30.2, 31.5, 33.0, 22.5, 26.8, 29.5],
        "Glukosa": [90, 95, 105, 120, 140, 155, 165, 92, 115, 145],
        "Tekanan Darah": [110, 115, 120, 125, 135, 140, 145, 112, 122, 130],
        "Riwayat Keluarga": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
        "Kelas": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    }
)


def muat_dataset(file) -> pd.DataFrame:
    """Baca CSV unggahan. Kolom wajib: FITUR + Kelas (0/1 atau Rendah/Tinggi)."""
    df = pd.read_csv(file)
    wajib = FITUR + ["Kelas"]
    kurang = [c for c in wajib if c not in df.columns]
    if kurang:
        raise ValueError(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
    if df["Kelas"].dtype == object:
        df["Kelas"] = df["Kelas"].str.strip().str.lower().map({"rendah": 0, "tinggi": 1})
    if df["Kelas"].isna().any() or not set(df["Kelas"].unique()) <= {0, 1}:
        raise ValueError("Kolom Kelas harus berisi 0/1 atau Rendah/Tinggi.")
    if "Pasien" not in df.columns:
        df.insert(0, "Pasien", [f"P{i + 1:02d}" for i in range(len(df))])
    return df[["Pasien"] + wajib].reset_index(drop=True)


# ----------------------------------------------------------------------------
# KNN
# ----------------------------------------------------------------------------
def normalisasi(X: pd.DataFrame, baru: pd.DataFrame, metode: str):
    """Kembalikan (X_norm, baru_norm, tabel_min_max)."""
    mn, mx = X.min(), X.max()
    if metode == "Min-Max (0-1)":
        rng = (mx - mn).replace(0, 1)
        return (X - mn) / rng, (baru - mn) / rng, pd.DataFrame({"Min": mn, "Max": mx})
    return X.copy(), baru.copy(), pd.DataFrame({"Min": mn, "Max": mx})


def hitung_jarak(Xn: pd.DataFrame, bn: pd.Series, metrik: str) -> np.ndarray:
    selisih = Xn.values - bn.values
    if metrik == "Euclidean":
        return np.sqrt((selisih**2).sum(axis=1))
    return np.abs(selisih).sum(axis=1)  # Manhattan


def prediksi(df, baru, k, metode, metrik):
    X = df[FITUR].astype(float)
    Xn, bn, tabel_mm = normalisasi(X, baru[FITUR].astype(float), metode)
    jarak = hitung_jarak(Xn, bn.iloc[0], metrik)

    hasil = df.copy()
    hasil["Jarak"] = jarak
    hasil = hasil.sort_values("Jarak").reset_index(drop=True)
    hasil.insert(0, "Peringkat", np.arange(1, len(hasil) + 1))

    tetangga = hasil.head(k)
    suara = tetangga["Kelas"].value_counts().reindex([0, 1], fill_value=0)
    if suara[0] == suara[1]:  # seri -> ikuti tetangga terdekat
        kelas = int(tetangga.iloc[0]["Kelas"])
    else:
        kelas = int(suara.idxmax())
    return kelas, suara, hasil, Xn, bn, tabel_mm


def loocv(df, k, metode, metrik):
    """Leave-One-Out Cross Validation untuk mengevaluasi nilai K."""
    benar = 0
    for i in range(len(df)):
        latih = df.drop(index=i).reset_index(drop=True)
        uji = df.iloc[[i]]
        kls, *_ = prediksi(latih, uji, min(k, len(latih)), metode, metrik)
        benar += int(kls == uji["Kelas"].iloc[0])
    return benar / len(df)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan Model")

sumber = st.sidebar.radio("Sumber dataset", ["Dataset studi kasus", "Unggah CSV"])
if sumber == "Unggah CSV":
    file = st.sidebar.file_uploader(
        "CSV dengan kolom: " + ", ".join(FITUR) + ", Kelas", type="csv"
    )
    try:
        data = muat_dataset(file) if file else DATA_DEFAULT.copy()
        if not file:
            st.sidebar.info("Belum ada file, memakai dataset studi kasus.")
    except Exception as e:
        st.sidebar.error(str(e))
        data = DATA_DEFAULT.copy()
else:
    data = DATA_DEFAULT.copy()

k = st.sidebar.slider("Nilai K", 1, max(1, len(data) - 1), min(3, len(data) - 1))
if k % 2 == 0:
    st.sidebar.warning("K genap berpotensi seri. Disarankan memakai K ganjil.")
metode_norm = st.sidebar.selectbox("Normalisasi", ["Min-Max (0-1)", "Tanpa normalisasi"])
metrik = st.sidebar.selectbox("Metrik jarak", ["Euclidean", "Manhattan"])

st.sidebar.caption(
    "Tanpa normalisasi, atribut berskala besar (Glukosa, Tekanan Darah) akan "
    "mendominasi perhitungan jarak."
)

# ----------------------------------------------------------------------------
# Header + Input pasien baru
# ----------------------------------------------------------------------------
st.title("🩺 SPK Prediksi Risiko Diabetes")
st.caption("Metode: K-Nearest Neighbors (KNN) · Target: 0 = Risiko Rendah, 1 = Risiko Tinggi")

st.subheader("1. Data Pasien Baru")
c1, c2, c3, c4, c5 = st.columns(5)
usia = c1.number_input("Usia (tahun)", 1, 120, 42)
bmi = c2.number_input("BMI", 10.0, 60.0, 28.5, step=0.1)
glukosa = c3.number_input("Glukosa (mg/dL)", 40, 400, 130)
tekanan = c4.number_input("Tekanan Darah (mmHg)", 60, 250, 128)
riwayat = c5.selectbox(
    "Riwayat Keluarga", [0, 1], index=1, format_func=lambda x: "Ada" if x else "Tidak ada"
)

pasien_baru = pd.DataFrame(
    [[usia, bmi, glukosa, tekanan, riwayat]], columns=FITUR
)

# ----------------------------------------------------------------------------
# Proses
# ----------------------------------------------------------------------------
kelas, suara, hasil, Xn, bn, tabel_mm = prediksi(data, pasien_baru, k, metode_norm, metrik)
tetangga = hasil.head(k)

tab_hasil, tab_hitung, tab_viz, tab_eval, tab_data = st.tabs(
    ["📋 Hasil Keputusan", "🧮 Langkah Perhitungan", "📊 Visualisasi", "🎯 Evaluasi K", "🗂️ Dataset"]
)

# --- Hasil -------------------------------------------------------------------
with tab_hasil:
    st.subheader("2. Hasil Klasifikasi")
    kiri, kanan = st.columns([1, 2])
    with kiri:
        if kelas == 1:
            st.error(f"### ⚠️ Risiko {LABEL[kelas].upper()}")
        else:
            st.success(f"### ✅ Risiko {LABEL[kelas].upper()}")
        m1, m2 = st.columns(2)
        m1.metric("Suara Rendah (0)", int(suara[0]))
        m2.metric("Suara Tinggi (1)", int(suara[1]))
        st.progress(float(suara[1]) / k, text=f"Proporsi tetangga berisiko tinggi: {suara[1]}/{k}")

    with kanan:
        st.markdown(f"**{k} tetangga terdekat** (jarak {metrik}, normalisasi: {metode_norm})")
        tampil = tetangga[["Peringkat", "Pasien"] + FITUR + ["Jarak", "Kelas"]].copy()
        tampil["Kelas"] = tampil["Kelas"].map(LABEL)
        st.dataframe(
            tampil.style.format({"Jarak": "{:.4f}", "BMI": "{:.1f}"}),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("3. Rekomendasi")
    if kelas == 1:
        st.markdown(
            "- Pasien memiliki profil kesehatan yang mirip dengan pasien **berisiko tinggi**.\n"
            "- Disarankan pemeriksaan lanjutan (HbA1c / glukosa puasa) dan konsultasi dokter.\n"
            "- Anjurkan pengaturan pola makan, aktivitas fisik, dan pemantauan tekanan darah."
        )
    else:
        st.markdown(
            "- Pasien memiliki profil kesehatan yang mirip dengan pasien **berisiko rendah**.\n"
            "- Pertahankan gaya hidup sehat dan lakukan pemeriksaan rutin berkala."
        )
    st.caption(
        "⚠️ Sistem ini hanya alat bantu keputusan berbasis data contoh yang sangat kecil, "
        "bukan pengganti diagnosis tenaga medis."
    )

# --- Langkah perhitungan -----------------------------------------------------
with tab_hitung:
    st.markdown("#### Langkah 1 - Nilai minimum & maksimum tiap atribut (dari data latih)")
    st.dataframe(tabel_mm.T, use_container_width=True)

    if metode_norm == "Min-Max (0-1)":
        st.markdown("#### Langkah 2 - Normalisasi Min-Max")
        st.latex(r"x' = \frac{x - x_{min}}{x_{max} - x_{min}}")
        st.markdown("Pasien baru setelah dinormalisasi:")
        st.dataframe(bn.to_frame().T.style.format("{:.4f}"), hide_index=True, use_container_width=True)
        st.markdown("Data latih setelah dinormalisasi:")
        tn = Xn.copy()
        tn.insert(0, "Pasien", data["Pasien"])
        st.dataframe(tn.style.format({c: "{:.4f}" for c in FITUR}), hide_index=True, use_container_width=True)
    else:
        st.info("Normalisasi dinonaktifkan - jarak dihitung dari nilai asli.")

    st.markdown("#### Langkah 3 - Hitung jarak ke setiap pasien")
    if metrik == "Euclidean":
        st.latex(r"d(p,q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}")
    else:
        st.latex(r"d(p,q) = \sum_{i=1}^{n}|p_i - q_i|")

    rinci = pd.DataFrame(
        {
            "Pasien": data["Pasien"],
            **{f"Δ {f}": (Xn[f] - bn[f]).abs() for f in FITUR},
            "Jarak": hitung_jarak(Xn, bn, metrik),
            "Kelas": data["Kelas"].map(LABEL),
        }
    )
    st.dataframe(
        rinci.style.format({c: "{:.4f}" for c in rinci.columns if c.startswith("Δ")} | {"Jarak": "{:.4f}"}),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(f"#### Langkah 4 - Urutkan jarak & ambil K = {k} terdekat")
    urut = hasil[["Peringkat", "Pasien", "Jarak", "Kelas"]].copy()
    urut["Kelas"] = urut["Kelas"].map(LABEL)
    urut["Dipilih"] = np.where(urut["Peringkat"] <= k, "✔", "")
    st.dataframe(urut.style.format({"Jarak": "{:.4f}"}), hide_index=True, use_container_width=True)

    st.markdown("#### Langkah 5 - Voting mayoritas")
    st.write(f"Rendah (0): **{suara[0]}** suara · Tinggi (1): **{suara[1]}** suara")
    st.write(f"➡️ Pasien baru diklasifikasikan sebagai **Risiko {LABEL[kelas]}**.")

# --- Visualisasi -------------------------------------------------------------
with tab_viz:
    st.markdown("Pilih dua atribut untuk melihat posisi pasien baru terhadap data latih.")
    v1, v2 = st.columns(2)
    sumbu_x = v1.selectbox("Sumbu X", FITUR, index=2)
    sumbu_y = v2.selectbox("Sumbu Y", FITUR, index=1)

    latih = data.copy()
    latih["Kelas_label"] = latih["Kelas"].map(LABEL)
    latih["Tetangga"] = latih["Pasien"].isin(tetangga["Pasien"])
    baru_plot = pasien_baru.copy()
    baru_plot["Pasien"] = "Pasien Baru"

    warna = alt.Scale(domain=["Rendah", "Tinggi"], range=["#2e8b57", "#d62728"])
    titik = (
        alt.Chart(latih)
        .mark_circle(size=140, opacity=0.85)
        .encode(
            x=alt.X(sumbu_x, scale=alt.Scale(zero=False)),
            y=alt.Y(sumbu_y, scale=alt.Scale(zero=False)),
            color=alt.Color("Kelas_label:N", scale=warna, title="Kelas"),
            stroke=alt.condition(alt.datum.Tetangga, alt.value("black"), alt.value("transparent")),
            strokeWidth=alt.value(2.5),
            tooltip=["Pasien"] + FITUR + ["Kelas_label"],
        )
    )
    label = (
        alt.Chart(latih)
        .mark_text(dy=-14, fontSize=11)
        .encode(x=sumbu_x, y=sumbu_y, text="Pasien")
    )
    bintang = (
        alt.Chart(baru_plot)
        .mark_point(shape="diamond", size=300, filled=True, color="#1f77b4")
        .encode(x=sumbu_x, y=sumbu_y, tooltip=["Pasien"] + FITUR)
    )
    st.altair_chart((titik + label + bintang).properties(height=450), use_container_width=True)
    st.caption("◆ biru = pasien baru · lingkaran bergaris hitam = K tetangga terdekat")

# --- Evaluasi K --------------------------------------------------------------
with tab_eval:
    st.markdown(
        "Akurasi **Leave-One-Out Cross Validation** untuk beberapa nilai K "
        "(setiap pasien diuji dengan model yang dilatih dari pasien lainnya)."
    )
    maks_k = min(9, len(data) - 1)
    daftar_k = list(range(1, maks_k + 1))
    akurasi = [loocv(data.reset_index(drop=True), kk, metode_norm, metrik) for kk in daftar_k]
    df_ak = pd.DataFrame({"K": daftar_k, "Akurasi": akurasi})

    grafik = (
        alt.Chart(df_ak)
        .mark_line(point=True)
        .encode(
            x=alt.X("K:O"),
            y=alt.Y("Akurasi:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            tooltip=["K", alt.Tooltip("Akurasi:Q", format=".0%")],
        )
        .properties(height=300)
    )
    st.altair_chart(grafik, use_container_width=True)
    terbaik = df_ak.loc[df_ak["Akurasi"].idxmax()]
    st.write(f"K terbaik pada pengaturan saat ini: **K = {int(terbaik['K'])}** (akurasi {terbaik['Akurasi']:.0%}).")
    st.caption("Dengan hanya 10 data, hasil evaluasi ini belum dapat digeneralisasi.")

# --- Dataset -----------------------------------------------------------------
with tab_data:
    tampil_data = data.copy()
    tampil_data["Kelas"] = tampil_data["Kelas"].map(LABEL)
    st.dataframe(tampil_data, hide_index=True, use_container_width=True)
    st.download_button(
        "⬇️ Unduh dataset (CSV)",
        data.to_csv(index=False).encode("utf-8"),
        "dataset_diabetes.csv",
        "text/csv",
    )
