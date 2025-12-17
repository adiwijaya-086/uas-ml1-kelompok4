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
.sidebar-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 1rem;
    border-bottom: 3px solid #b0d9cc;
    padding-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
body { background-color: #eaf4f2; } 

/* Hero Section */
.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #2ecc71, #3498db);
    color: white;
    margin-bottom: 2rem;
}
.hero h1 { font-size: 2.5rem; }

/* Card Style */
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 18px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}

/* Footer */
.footer {
    text-align:center;
    color:gray;
    font-size:0.9rem;
    margin-top:2rem;
}

/* Sidebar Flat Style */
.sidebar .sidebar-content {
    background-color:#eaf4f2;
    padding: 20px;
}
.sidebar-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 1rem;
}
button[kind="primary"] {
    background-color: transparent;
    color: #333;
    font-size: 16px;
    border: none;
    text-align: left;
    width: 100%;
    padding: 0.6rem 0;
    transition: background-color 0.3s ease;
}
button[kind="primary"]:hover {
    background-color: #e0f7ef;
    color: #2ecc71;
    font-weight: 600;
}
button[kind="primary"][aria-pressed="true"] {
    background-color: #d1f2eb;
    color: #27ae60;
    font-weight: bold;
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
# SIDEBAR NAVIGASI
# ===============================
st.sidebar.markdown('<div class="sidebar-title">Dashboard</div>', unsafe_allow_html=True)

if "menu" not in st.session_state:
    st.session_state.menu = "Beranda"

for label in ["Beranda", "Data & EDA", "Clustering", "Peta Cluster", "Cari Kabupaten"]:
    if st.sidebar.button(label, key=label):
        st.session_state.menu = label

menu = st.session_state.menu

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

# ===============================
# CLUSTER DESCRIPTION
# ===============================
cluster_desc = {
    0: {"judul": "Pengelolaan Rendah", "deskripsi": "Wilayah dengan volume sampah rendah dan kebutuhan armada minimal."},
    1: {"judul": "Pengelolaan Sedang", "deskripsi": "Daerah dengan tingkat produksi sampah sedang dan distribusi armada yang cukup."},
    2: {"judul": "Pengelolaan Tinggi", "deskripsi": "Wilayah dengan intensitas produksi sampah tinggi dan kebutuhan armada besar."}
}

# ===============================
# PAGE CONTENT BASED ON MENU
# ===============================
if menu == "Beranda":
    st.markdown("### 🎯 Tujuan Aplikasi")
    st.write("Mengelompokkan kabupaten/kota di Jawa Barat berdasarkan karakteristik pengelolaan sampah menggunakan **K-Means Clustering**.")
    st.markdown("### ⚙️ Metodologi")
    st.write("- Normalisasi data (StandardScaler)\n- Reduksi dimensi menggunakan PCA\n- Clustering menggunakan K-Means\n- Visualisasi spasial menggunakan peta interaktif")
    st.markdown("### 📰 Berita Terkini")

    berita = [
        {
            "judul": "Volume Sampah di Jawa Barat Tembus 29 Ribu Ton per Hari, Jadi Tantangan Besar Pengelolaan",
            "img": "https://asset.kompas.com/crops/LzE1yzwz7Lhg7HJTiv5KvswqCy8=/0x0:0x0/1200x800/data/photo/2025/07/31/688b7374ce1dc.jpeg",
            "ringkasan": "Sampah di Jawa Barat kini mencapai sekitar 29 ribu ton per hari, menuntut solusi pengelolaan yang lebih efektif..",
            "url": "https://www.bing.com/ck/a?!&&p=1bb5867cad6c24dd1f64a3ea4b671b90d5fec93cc9d617664a5678c3609bb535JmltdHM9MTc2NTkyOTYwMA&ptn=3&ver=2&hsh=4&fclid=184463fe-eb14-6ea0-172b-7078ea496fe0&psq=Sekda+Jabar+ungkap+timbunan+sampah+mencapai+29%2c7+ribu+ton+per+hari.&u=a1aHR0cHM6Ly9iYW5kdW5nLmtvbXBhcy5jb20vcmVhZC8yMDI1LzA4LzAxLzA1NDAwMzM3OC91bmdrYXAtc2FtcGFoLWRpLWphYmFyLXRlbWJ1cy0yOS1yaWJ1LXRvbi1wZXItaGFyaS1zZWtkYS15YW5nLW1lbnllZGloa2FuIzp-OnRleHQ9U1VLQUJVTUklMkMlMjBLT01QQVMuY29tJTIwLSUyMFNla3JldGFyaXMlMjBEYWVyYWglMjAlMjhTZWtkYSUyOSUyMFByb3ZpbnNpJTIwSmF3YSxKYWJhciUyMHRlbWJ1cyUyMGhpbmdnYSUyMDI5JTIwcmlidSUyMHRvbiUyMHBlciUyMGhhcmlueWEu"
        },
        {
            "judul": "Dedi Mulyadi Turun ke Sungai Cipalabuhan untuk Bersihkan Sampah yang Menyumbat Aliran Air",
            "img": "https://asset.kompas.com/crops/nxpuRDSR0zaQwsamjWmBFJhj9Nc=/0x0:1280x853/1200x800/data/photo/2025/03/08/67cc48d363b10.jpg",
            "ringkasan": "Dedi Mulyadi turun langsung ke sungai untuk membersihkan sampah dan menyoroti kerusakan lingkungan di Jawa Barat.",
            "url": "hhhttps://www.kompas.com/jawa-barat/read/2025/03/08/204314988/turun-ke-sungai-bersihkan-sampah-dedi-mulyadi-hutan-dirusak-malah"
        },
        {
            "judul": "Tingginya Produksi Sampah Harian di Jawa Barat Jadi Tantangan Serius Pengelolaan Lingkungan",
            "img": "https://images.bisnis.com/posts/2023/05/09/1654204/screenshot_20230509-161253_photopictureresizer_copy_1000x667.jpg",
            "ringkasan": "Jawa Barat menghasilkan puluhan ribu ton sampah per hari, namun pengelolaannya belum optimal.",
            "url": "https://bapenda.jabarprov.go.id/2025/10/09/gubernur-jawa-barat-umumkan-pembangunan-pembangkit-listrik-tenaga-sampah-di-seluruh-wilayah-jabar/"
        }
    ]

    cols = st.columns(3)
    for i, b in enumerate(berita):
        with cols[i % 3]:
            st.image(b["img"], caption=b["judul"], use_container_width=True)
            st.write(b["ringkasan"])
            st.markdown(f"[Baca selengkapnya]({b['url']})")

elif menu == "Data & EDA":
    tahun = st.selectbox("**Pilih Tahun**", [2020, 2021, 2022, 2023])
    df = load_data(tahun)
    st.dataframe(df)
    st.subheader("📊 Statistik Deskriptif")
    st.dataframe(df.describe())
    st.subheader("Total Sampah per Kabupaten")
    st.bar_chart(df.set_index("kabupaten")["total_sampah_ton"])

elif menu == "Clustering":
    tahun = st.selectbox("**Pilih Tahun Clustering**", [2020, 2021, 2022, 2023])
    df = load_data(tahun)
    kmeans, scaler, pca = load_model(tahun)

    features = [
        'jumlah_penduduk','total_sampah_ton','jumlah_truk','jumlah_motor',
        'jumlah_tps','total_armada','sampah_perpenduduk',
        'sampah_perarmada','sampah_pertps'
    ]
    X = df[features]
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    df["pc1"] = X_pca[:,0]
    df["pc2"] = X_pca[:,1]
    df["cluster"] = kmeans.predict(X_pca)

    fig = px.scatter(df, x="pc1", y="pc2",
                     color=df["cluster"].astype(str),
                     hover_data=["kabupaten"],
                     title=f"Visualisasi PCA + K-Means ({tahun})")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[["kabupaten","pc1","pc2","cluster"]].sort_values("cluster"))

elif menu == "Peta Cluster":
    tahun = st.selectbox("**Pilih Tahun Peta**", [2020, 2021, 2022, 2023])
    df = load_data(tahun)
    kmeans, scaler, pca = load_model(tahun)
    X = df[['jumlah_penduduk','total_sampah_ton','jumlah_truk','jumlah_motor',
            'jumlah_tps','total_armada','sampah_perpenduduk','sampah_perarmada','sampah_pertps']]
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    df["cluster"] = kmeans.predict(X_pca)

    geo = gpd.read_file("geo/jabar_kabupaten.geojson")
    geo["KABKOT"] = geo["KABKOT"].str.lower().str.strip()
    df["kabupaten"] = df["kabupaten"].str.lower().str.strip()
    geo = geo.merge(df[["kabupaten","cluster"]], left_on="KABKOT", right_on="kabupaten", how="left")

    m = folium.Map(location=[-6.9,107.6], zoom_start=8, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=geo,
        data=geo,
        columns=["kabupaten","cluster"],
        key_on="feature.properties.KABKOT",
        fill_color="Set1", fill_opacity=0.75, line_opacity=0.3,
        legend_name="Cluster Kabupaten"
    ).add_to(m)

    for _,r in geo.iterrows():
        folium.Marker(
            location=[r.geometry.centroid.y, r.geometry.centroid.x],
            popup=f"{r['KABKOT'].title()} - Cluster {r['cluster']}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    st_folium(m, width=1200, height=550)

elif menu == "Cari Kabupaten":
    opsi_tahun = st.radio("**Pilih Mode Tampilan**", ["Pilih Tahun", "Tampilkan Semua Tahun"])

    if opsi_tahun == "Pilih Tahun":
        tahun_cari = st.selectbox("Tahun", [2020, 2021, 2022, 2023])
        df_cari = load_data(tahun_cari)
        kmeans_cari, scaler_cari, pca_cari = load_model(tahun_cari)

        fitur = ['jumlah_penduduk','total_sampah_ton','jumlah_truk','jumlah_motor',
                 'jumlah_tps','total_armada','sampah_perpenduduk','sampah_perarmada','sampah_pertps']
        X_cari = df_cari[fitur]
        X_scaled_cari = scaler_cari.transform(X_cari)
        X_pca_cari = pca_cari.transform(X_scaled_cari)
        df_cari["cluster"] = kmeans_cari.predict(X_pca_cari)

        pilih = st.selectbox("🔍 Pilih Kabupaten", sorted(df_cari["kabupaten"].unique()))
        hasil = df_cari[df_cari["kabupaten"] == pilih]
        cluster_id = int(hasil["cluster"].values[0])
        info = cluster_desc.get(cluster_id, {
            "judul": "Tidak Dikenal",
            "deskripsi": "Belum ada deskripsi untuk cluster ini."
        })

        st.markdown(f"""
        <div style="background-color:#eaf4f2;padding:1.5rem;border-radius:12px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:1.5rem;">
            <h3 style="margin-bottom:0.5rem;">📍 <b>{pilih.title()}</b> ({tahun_cari})</h3>
            <p style="font-size:1.1rem;margin-bottom:0.5rem;">
                <b>Cluster {cluster_id} — {info['judul']}</b>
            </p>
            <p style="line-height:1.6;font-size:0.95rem;">
                {info['deskripsi']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah TPS", int(hasil["jumlah_tps"].values[0]))
        col2.metric("Sampah per Armada", f"{hasil['sampah_perarmada'].values[0]:.2f}")
        col3.metric("Total Sampah (ton)", f"{float(hasil['total_sampah_ton'].values[0]):,.2f}")

        st.markdown("### 📊 Data Lengkap")
        st.dataframe(hasil)

    else:
        semua_data = []
        for tahun_loop in [2020, 2021, 2022, 2023]:
            df_loop = load_data(tahun_loop)
            kmeans_loop, scaler_loop, pca_loop = load_model(tahun_loop)

            fitur = ['jumlah_penduduk','total_sampah_ton','jumlah_truk','jumlah_motor',
                    'jumlah_tps','total_armada','sampah_perpenduduk','sampah_perarmada','sampah_pertps']
            X_loop = df_loop[fitur]
            X_scaled_loop = scaler_loop.transform(X_loop)
            X_pca_loop = pca_loop.transform(X_scaled_loop)
            df_loop["cluster"] = kmeans_loop.predict(X_pca_loop)
            df_loop["tahun"] = tahun_loop
            semua_data.append(df_loop)

        gabung = pd.concat(semua_data)
        pilih = st.selectbox("🔍 Pilih Kabupaten", sorted(gabung["kabupaten"].unique()))
        hasil_gabung = gabung[gabung["kabupaten"] == pilih]

        st.markdown(f"### 📍 {pilih.title()} dari Tahun 2020–2023")

        for _, row in hasil_gabung.iterrows():
            cid = int(row["cluster"])
            info = cluster_desc.get(cid, {"judul": "Tidak Dikenal", "deskripsi": "Belum ada deskripsi."})

            st.markdown(f"""
            <div style="background-color:#f0f9f7;padding:1rem;border-radius:10px;
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
        st.dataframe(hasil_gabung)

        cluster_desc = {
    0: {
        "judul": "Pengelolaan Rendah",
        "deskripsi": "Wilayah dengan volume sampah rendah dan jumlah armada terbatas. Cocok untuk pendekatan pengelolaan sederhana dan efisien, dengan fokus pada edukasi masyarakat dan optimalisasi TPS."
    },
    1: {
        "judul": "Pengelolaan Sedang",
        "deskripsi": "Kabupaten dengan tingkat produksi sampah sedang dan distribusi armada yang cukup. Perlu strategi pengelolaan menengah yang seimbang antara operasional dan edukasi lingkungan."
    },
    2: {
        "judul": "Pengelolaan Tinggi",
        "deskripsi": "Wilayah dengan intensitas produksi sampah tinggi dan kebutuhan armada besar. Biasanya mencerminkan area urban padat yang memerlukan sistem pengelolaan kompleks dan terintegrasi."
    }
}
st.markdown("""
<div class="footer">
    Dibuat untuk UAS Machine Learning | K-Means Clustering + PCA | Jawa Barat <br>
    GitHub: <a href="https://github.com/adiwijaya-086" target="_blank">SampahKita123</a>
</div>
""", unsafe_allow_html=True)
