import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="SampahKita | Analisis Clustering Sampah Jabar",
    page_icon="♻️",
    layout="wide"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.hero {
    padding: 3rem;
    border-radius: 25px;
    background: linear-gradient(135deg, #2ecc71, #3498db);
    color: white;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 3rem;
}
.hero p {
    font-size: 1.1rem;
    max-width: 900px;
}
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 18px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 0.9rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HERO SECTION
# ===============================
st.markdown("""
<div class="hero">
    <h1>♻️ SampahKita</h1>
    <h3>Analisis Clustering Pengelolaan Sampah Kabupaten Jawa Barat</h3>
    <p>
    Aplikasi analitik berbasis <b>Machine Learning</b> menggunakan
    <b>K-Means Clustering</b> dan <b>Principal Component Analysis (PCA)</b>
    untuk membantu pengambilan keputusan pengelolaan sampah berkelanjutan
    di Jawa Barat.
    </p>
</div>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("📊 Panel Data")
tahun = st.sidebar.selectbox("📅 Pilih Tahun Analisis", [2020, 2021, 2022, 2023])

# ===============================
# LOAD DATA & MODEL
# ===============================
@st.cache_data
def load_data(tahun):
    df = pd.read_csv(f"data/data{tahun}.csv", sep=";", decimal=",")
    df.columns = df.columns.str.lower().str.strip()
    return df

@st.cache_resource
def load_model(tahun):
    folder = f"model{tahun}"
    kmeans = pickle.load(open(f"{folder}/kmeans_{tahun}.pkl", "rb"))
    scaler = pickle.load(open(f"{folder}/scaler_{tahun}.pkl", "rb"))
    pca = pickle.load(open(f"{folder}/pca_{tahun}.pkl", "rb"))
    return kmeans, scaler, pca

df = load_data(tahun)
kmeans, scaler, pca = load_model(tahun)

# ===============================
# FEATURE ENGINEERING
# ===============================
features = [
    'jumlah_penduduk', 'total_sampah_ton', 'jumlah_truk', 'jumlah_motor',
    'jumlah_tps', 'total_armada', 'sampah_perpenduduk',
    'sampah_perarmada', 'sampah_pertps'
]

X = df[features]
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

df["pc1"] = X_pca[:, 0]
df["pc2"] = X_pca[:, 1]
df["cluster"] = kmeans.predict(X_pca)

# ===============================
# SIDEBAR INSIGHT
# ===============================
st.sidebar.markdown(f"### 📌 Ringkasan Tahun {tahun}")
st.sidebar.metric("Jumlah Kabupaten", len(df))
st.sidebar.metric("Rata-rata Sampah/Armada", f"{df['sampah_perarmada'].mean():.2f}")

st.sidebar.markdown("### 🔍 Cari Kabupaten")
kabupaten_pilih = st.sidebar.selectbox("Kabupaten", sorted(df["kabupaten"].unique()))
hasil_sidebar = df[df["kabupaten"] == kabupaten_pilih]
st.sidebar.success(f"{kabupaten_pilih.title()} → Cluster {int(hasil_sidebar['cluster'].values[0])}")

# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Beranda",
    "📊 Data & EDA",
    "🧠 Clustering",
    "🗺️ Peta Cluster",
    "🔍 Cari Kabupaten"
])

# ===============================
# TAB 1 — BERANDA
# ===============================
with tab1:
    st.markdown("""
    ### 🎯 Tujuan Aplikasi
    Mengelompokkan kabupaten/kota di Jawa Barat berdasarkan karakteristik pengelolaan sampah menggunakan **K-Means Clustering**.

    ### ⚙️ Metodologi
    - Normalisasi data (StandardScaler)
    - Reduksi dimensi menggunakan PCA
    - Clustering menggunakan K-Means
    - Visualisasi spasial menggunakan peta interaktif
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# TAB 2 — EDA
# ===============================
with tab2:
    with st.expander("📊 Statistik Deskriptif"):
        st.dataframe(df[features].describe())

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Total Sampah per Kabupaten")
    st.bar_chart(df.set_index("kabupaten")["total_sampah_ton"])
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# TAB 3 — CLUSTERING
# ===============================
with tab3:
    st.markdown("## 🧠 Visualisasi Clustering per Tahun")

    # Pilih tahun untuk visualisasi
    tahun_cluster = st.selectbox("📅 Pilih Tahun Clustering", [2020, 2021, 2022, 2023])
    df_c = load_data(tahun_cluster)
    kmeans_c, scaler_c, pca_c = load_model(tahun_cluster)

    # Proses clustering ulang
    X_c = df_c[features]
    X_scaled_c = scaler_c.transform(X_c)
    X_pca_c = pca_c.transform(X_scaled_c)
    df_c["pc1"] = X_pca_c[:, 0]
    df_c["pc2"] = X_pca_c[:, 1]
    df_c["cluster"] = kmeans_c.predict(X_pca_c)

    # Visualisasi PCA + K-Means
    fig = px.scatter(
        df_c, x="pc1", y="pc2",
        color=df_c["cluster"].astype(str),
        hover_data=["kabupaten"],
        title=f"Visualisasi PCA + K-Means ({tahun_cluster})"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tampilkan DataFrame hasil clustering
    st.markdown("### 📊 Hasil Clustering")
    st.dataframe(df_c[["kabupaten", "pc1", "pc2", "cluster"]].sort_values("cluster"))

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# TAB 4 — MAP
# ===============================
with tab4:
    geo = gpd.read_file("geo/jabar_kabupaten.geojson")
    geo["KABKOT"] = geo["KABKOT"].str.lower().str.strip()
    df["kabupaten"] = df["kabupaten"].str.lower().str.strip()

    geo = geo.merge(
        df[["kabupaten", "cluster"]],
        left_on="KABKOT",
        right_on="kabupaten",
        how="left"
    )

    m = folium.Map(location=[-6.9, 107.6], zoom_start=8, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=geo,
        data=geo,
        columns=["kabupaten", "cluster"],
        key_on="feature.properties.KABKOT",
        fill_color="Set1",
        fill_opacity=0.75,
        line_opacity=0.3,
        legend_name="Cluster Kabupaten"
    ).add_to(m)

    for _, r in geo.iterrows():
        folium.Marker(
            location=[r.geometry.centroid.y, r.geometry.centroid.x],
            popup=f"{r['KABKOT'].title()} - Cluster {r['cluster']}"
        ).add_to(m)

    st_folium(m, width=1200, height=550)
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# CLUSTER DESCRIPTION
# ===============================
cluster_desc = {
    0: {
        "judul": "Pengelolaan Rendah",
        "deskripsi": "Wilayah dengan volume sampah rendah dan kebutuhan armada minimal. Cocok untuk pendekatan pengelolaan sederhana."
    },
    1: {
        "judul": "Pengelolaan Sedang",
        "deskripsi": "Daerah dengan tingkat produksi sampah sedang dan distribusi armada yang cukup. Cocok untuk strategi pengelolaan menengah."
    },
    2: {
        "judul": "Pengelolaan Tinggi",
        "deskripsi": "Wilayah dengan intensitas produksi sampah tinggi dan kebutuhan armada besar. Umumnya mencerminkan area urban padat dan aktif."
    }
}

# ===============================
# TAB 5 — SEARCH
# ===============================
with tab5:
    st.markdown("## 🔍 Sampah Kabupaten per Tahun")

    # Opsi tampilan: satu tahun atau semua tahun
    opsi_tahun = st.radio("📅 Pilih Mode Tampilan", ["Pilih Tahun", "Tampilkan Semua Tahun"])
    pilih = st.selectbox("🔍 Pilih Kabupaten", sorted(df["kabupaten"].unique()))

    if opsi_tahun == "Pilih Tahun":
        tahun_cari = st.selectbox("Tahun", [2020, 2021, 2022, 2023])
        df_cari = load_data(tahun_cari)
        kmeans_cari, scaler_cari, pca_cari = load_model(tahun_cari)

        X_cari = df_cari[features]
        X_scaled_cari = scaler_cari.transform(X_cari)
        X_pca_cari = pca_cari.transform(X_scaled_cari)
        df_cari["pc1"] = X_pca_cari[:, 0]
        df_cari["pc2"] = X_pca_cari[:, 1]
        df_cari["cluster"] = kmeans_cari.predict(X_pca_cari)

        hasil = df_cari[df_cari["kabupaten"] == pilih]
        cluster_id = int(hasil["cluster"].values[0])
        info = cluster_desc.get(cluster_id, {
            "judul": "Kategori Tidak Dikenal",
            "deskripsi": "Deskripsi belum tersedia untuk cluster ini."
        })

        st.markdown(f"""
        <div style="background-color:#e6f4ea;padding:1rem;border-radius:10px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.1);margin-bottom:1rem;">
            <h4>📍 {pilih.title()} ({tahun_cari})</h4>
            <p><b>Cluster {cluster_id} — {info['judul']}</b></p>
            <p style="line-height:1.6;">{info['deskripsi']}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah TPS", int(hasil["jumlah_tps"].values[0]))
        col2.metric("Sampah per Armada", f"{hasil['sampah_perarmada'].values[0]:.2f}")
        col3.metric("Total Sampah (ton)", f"{float(hasil['total_sampah_ton'].values[0]):,.2f}")

        st.markdown("### 📊 Data Lengkap dengan Cluster")
        st.dataframe(hasil)

    else:
        semua_data = []
        for tahun_loop in [2020, 2021, 2022, 2023]:
            df_loop = load_data(tahun_loop)
            kmeans_loop, scaler_loop, pca_loop = load_model(tahun_loop)

            X_loop = df_loop[features]
            X_scaled_loop = scaler_loop.transform(X_loop)
            X_pca_loop = pca_loop.transform(X_scaled_loop)
            df_loop["pc1"] = X_pca_loop[:, 0]
            df_loop["pc2"] = X_pca_loop[:, 1]
            df_loop["cluster"] = kmeans_loop.predict(X_pca_loop)
            df_loop["tahun"] = tahun_loop

            hasil_loop = df_loop[df_loop["kabupaten"] == pilih]
            semua_data.append(hasil_loop)

        gabung = pd.concat(semua_data)

        st.markdown(f"### 📍 {pilih.title()} dari Tahun 2020–2023")
        for _, row in gabung.iterrows():
            cid = int(row["cluster"])
            info = cluster_desc.get(cid, {"judul": "", "deskripsi": ""})
            st.markdown(f"""
            <div style="background-color:#e6f4ea;padding:1rem;border-radius:10px;
                        box-shadow:0 2px 6px rgba(0,0,0,0.05);margin-bottom:1rem;">
                <h4>{row['tahun']}</h4>
                <p><b>Cluster {cid} — {info['judul']}</b></p>
                <p style="line-height:1.6;">{info['deskripsi']}</p>
                <ul>
                    <li>Jumlah TPS: {int(row['jumlah_tps'])}</li>
                    <li>Sampah per Armada: {row['sampah_perarmada']:.2f}</li>
                    <li>Total Sampah: {float(row['total_sampah_ton']):,.2f} ton</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Data Lengkap Semua Tahun")
        st.dataframe(gabung)

    st.markdown('</div>', unsafe_allow_html=True)
# ===============================
# FOOTER
# ===============================
st.markdown("""
<div class="footer">
    Dibuat untuk UAS Machine Learning | K-Means Clustering + PCA | Jawa Barat <br>
    GitHub: <a href="https://github.com/username/sampahkita" target="_blank">SampahKita</a>
</div>
""", unsafe_allow_html=True)