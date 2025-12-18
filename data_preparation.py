import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
import os

# ====================== PAGE CONFIG & STYLE ======================
st.set_page_config(page_title="Data Preparation - DES", layout="wide")

st.markdown("""
<style>
    .main {background-color: #0E1117; color: #E5E7EB;}
    .stApp {background-color: #0E1117;}
    h1, h2, h3, h4, h5, h6 {color: #00E396; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] button {color: #E5E7EB;}
    .info-box {
        background: #1a202c;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00E396;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ====================== TITLE ======================
st.markdown("# Data Preparation")
st.markdown("*Data Cleaning, Transformation, dan Exploration untuk Income Inequality South Africa*")

# ====================== LOAD DATA ======================
@st.cache_data
def load_raw_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Income Inequality in South Africa_Dataset.xlsx")
    df = pd.read_excel(file_path)
    return df

# Load dataset
df_original = load_raw_data().copy()

st.markdown("---")
st.markdown("## 📥 STEP 1: Data Loading & Initial Exploration")

st.markdown("""
<div class='info-box'>
<strong>📌 Penjelasan Step 1:</strong><br>
✓ Membaca file Excel yang berisi data Income Inequality South Africa<br>
✓ Menggunakan @st.cache_data untuk optimasi performa (data tidak reload setiap kali interaksi)<br>
✓ Menyimpan copy original untuk perbandingan sebelum vs sesudah preprocessing
</div>
""", unsafe_allow_html=True)

# Display data info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows", df_original.shape[0])
with col2:
    st.metric("Total Columns", df_original.shape[1])
with col3:
    st.metric("Missing Values", df_original.isnull().sum().sum())

# Preview data
st.subheader("Data Preview (First 5 Rows)")
st.dataframe(df_original.head(), use_container_width=True)

# Data Info
st.subheader("Data Information")
tabs1 = st.tabs(["Data Types", "Summary Statistics", "Missing Values"])

with tabs1[0]:
    st.write("**Column Data Types:**")
    info_df = pd.DataFrame({
        "Column": df_original.columns,
        "Data Type": df_original.dtypes.astype(str),
        "Non-Null Count": df_original.count(),
        "Null Count": df_original.isnull().sum()
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

with tabs1[1]:
    st.write("**Summary Statistics (Descriptive):**")
    st.dataframe(df_original.describe(), use_container_width=True)

with tabs1[2]:
    st.write("**Missing Values per Column:**")
    missing_df = pd.DataFrame({
        "Column": df_original.columns,
        "Missing Count": df_original.isnull().sum(),
        "Missing %": (df_original.isnull().sum() / len(df_original) * 100).round(2)
    })
    st.dataframe(missing_df[missing_df["Missing Count"] > 0], use_container_width=True, hide_index=True)

# Display full dataset
with st.expander("📊 View Full Dataset"):
    st.dataframe(df_original, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("## 🔧 STEP 2: Data Sorting & Interpolation")

st.markdown("""
<div class='info-box'>
<strong>📌 Penjelasan Step 2:</strong><br>
✓ Mengurutkan data berdasarkan Year (wajib untuk time series interpolation)<br>
✓ Mengidentifikasi kolom numerik (menghilangkan Year dari daftar interpolasi)<br>
✓ Melakukan Linear Interpolation untuk mengisi missing values dengan nilai yang proporsional antara dua data terdekat<br>
✓ Linear Interpolation cocok karena trend data yang smooth dan consistent
</div>
""", unsafe_allow_html=True)

# Sort dan Interpolasi
df_clean = df_original.sort_values(by='Year').reset_index(drop=True)

# Ambil kolom numerik selain Year
numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
if 'Year' in numeric_cols:
    numeric_cols.remove('Year')

# Lakukan interpolasi
for col in numeric_cols:
    df_clean[col] = df_clean[col].interpolate(method='linear')

st.success("✅ Data telah disort berdasarkan Year dan dilakukan interpolasi linear")

# Tampilkan hasil interpolasi
st.subheader("Data Setelah Interpolasi")
col1, col2 = st.columns(2)
with col1:
    st.write("**Missing Values Sebelum Interpolasi:**")
    st.dataframe(df_original.isnull().sum(), use_container_width=True)
with col2:
    st.write("**Missing Values Sesudah Interpolasi:**")
    st.dataframe(df_clean.isnull().sum(), use_container_width=True)

st.markdown("---")
st.markdown("## 📈 STEP 3: Visualisasi Interpolasi - Before vs After")

st.markdown("""
<div class='info-box'>
<strong>📌 Penjelasan Step 3:</strong><br>
✓ Membandingkan visualisasi data SEBELUM interpolasi (dengan missing values) vs SESUDAH<br>
✓ Garis merah (before) menunjukkan data asli dengan gaps pada missing values<br>
✓ Garis biru (after) menunjukkan hasil interpolasi yang smooth dan continuous<br>
✓ Membantu kita mengidentifikasi apakah interpolasi dilakukan dengan tepat
</div>
""", unsafe_allow_html=True)

# Pilih kolom untuk visualisasi interpolasi
selected_col_interp = st.selectbox("Pilih Kolom untuk Visualisasi Interpolasi:", numeric_cols)

fig, ax = plt.subplots(figsize=(12, 5))

# Plot sebelum interpolasi (original dengan missing values)
ax.plot(df_original['Year'], df_original[selected_col_interp],
        'o-', color='red', label='Before (Raw Data)', alpha=0.7, linewidth=2, markersize=8)

# Plot sesudah interpolasi
ax.plot(df_clean['Year'], df_clean[selected_col_interp],
        '-', color='#00E396', label='After Interpolation', linewidth=2.5)

ax.set_title(f"Perbandingan Interpolasi: {selected_col_interp}", fontsize=14, fontweight='bold')
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel(selected_col_interp, fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()

st.pyplot(fig)

st.markdown("---")
st.markdown("## 🎯 STEP 4: Column Selection & Filtering")

st.markdown("""
<div class='info-box'>
<strong>📌 Penjelasan Step 4:</strong><br>
✓ Memilih kolom yang relevan untuk analisis forecasting dan machine learning<br>
✓ Menghilangkan kolom yang tidak diperlukan (noise reduction)<br>
✓ Fokus pada variabel yang berkaitan dengan income inequality dan faktor-faktor ekonomi<br>
✓ Kolom yang dipilih harus memiliki data berkualitas dan tidak terlalu banyak missing values
</div>
""", unsafe_allow_html=True)

# Define selected columns
selected_cols = [
    'Year',
    'gini_disp',           # Gini - Disposable Income (target variable)
    'gini_mkt',            # Gini - Market Income
    'Inflation rate',      # Inflation rate
    'GDP',                 # Gross Domestic Product
    'GOVEDU',              # Government Education Spending
    'GOVEXP',              # Government Expenditure
    'FINDEV 1',            # Financial Development
    'DEMOCRACY',           # Democracy Index
    'FLABOUR'              # Labour Force
]

# Filter dataframe
df_filtered = df_clean[selected_cols].copy()

st.subheader("Kolom yang Dipilih untuk Analisis")
col_info = pd.DataFrame({
    "No": range(1, len(selected_cols) + 1),
    "Kolom": selected_cols,
    "Deskripsi": [
        "Tahun pengamatan",
        "Gini Coefficient (Disposable Income) - TARGET VARIABLE",
        "Gini Coefficient (Market Income)",
        "Inflation rate (%)",
        "Gross Domestic Product",
        "Government Education Spending",
        "Government Expenditure",
        "Financial Development Index",
        "Democracy Index",
        "Labour Force Participation"
    ]
})
st.dataframe(col_info, use_container_width=True, hide_index=True)

st.subheader("Data Hasil Filtering")
st.dataframe(df_filtered, use_container_width=True, hide_index=True)

# Summary
st.subheader("📊 Summary Statistik Filtered Data")
st.dataframe(df_filtered.describe(), use_container_width=True)

st.markdown("---")
st.markdown("## ✅ Data Preparation Complete!")

st.success("""
✓ Dari **{}** baris, **{}** kolom awal<br>
✓ Setelah filtering: **{}** baris, **{}** kolom<br>
✓ Missing values telah diatasi dengan interpolasi linear<br>
✓ Data siap untuk exploratory data analysis (EDA) dan modeling
""".format(
    df_original.shape[0],
    df_original.shape[1],
    df_filtered.shape[0],
    df_filtered.shape[1]
), icon="✅")

# # Download prepared data
# st.subheader("💾 Download Data yang Sudah Diproses")
# csv = df_filtered.to_csv(index=False)
# st.download_button(
#     label="📥 Download CSV",
#     data=csv,
#     file_name="data_prepared.csv",
#     mime="text/csv"
# )
