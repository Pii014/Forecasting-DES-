import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ====================== PAGE CONFIG & STYLE ======================
st.set_page_config(page_title="Income Inequality Forecast - DES", layout="wide")

st.markdown("""
<style>
    .main {background-color: #0E1117; color: #E5E7EB;}
    .stApp {background-color: #0E1117;}
    h1, h2, h3, h4, h5, h6 {color: #00E396; font-weight: bold;}
    .stTextArea label, .stNumberInput label, .stSelectbox label, .stSlider label {color: #E5E7EB !important;}
    .metric-card {
        background: linear-gradient(135deg, #1e242f, #2a3244);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.6);
        text-align: center;
        border: 1px solid #334155;
    }
    .info-card {
        background: #1a202c;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #00E396;
        height: 100%;
    }
    .stButton > button {
        background: #00E396 !important;
        color: black !important;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: #00ffb8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Income Inequality in South Africa_Dataset.xlsx")
    df = pd.read_excel(file_path)
    
    # Sort berdasarkan tahun (wajib biar interpolasinya benar)
    df = df.sort_values(by='Year')
    
    # Ambil hanya kolom numerik
    numeric_cols = df.select_dtypes(include='number').columns
    numeric_cols = [col for col in numeric_cols if col.lower() != 'year']
    
    # Interpolasi time series linear
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='linear')
    
    return df

df_raw = load_data()

# ====================== TITLE & HEADER ======================
st.markdown("# Income Inequality in South Africa - Gini Forecast")
st.markdown("*Forecasting Gini Coefficient menggunakan Double Exponential Smoothing*")

# ====================== DATA DISPLAY ======================
st.subheader("Data Lengkap - Income Inequality South Africa")
df_display = df_raw[['Year', 'gini_disp']].copy()
st.dataframe(df_display, use_container_width=True, hide_index=True)

st.markdown("---")

# ====================== SIDEBAR INPUT ======================
with st.sidebar:
    st.header("Parameter & Pengaturan")
    
    st.markdown("---")

    alpha = st.slider("Alpha (α)", min_value=0.01, max_value=0.99, value=0.60, step=0.01,
                      help="Semakin tinggi → semakin responsif terhadap data terbaru. Coba 0.1-0.3 untuk data stabil")

    periods_ahead = st.number_input("Periode Prediksi ke Depan (Tahun)", min_value=1, max_value=20, value=5, step=1)

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Hitung Forecast", type="primary", use_container_width=True):
            st.session_state.calculate = True
    with col_btn2:
        if st.button("Reset", use_container_width=True):
            st.session_state.calculate = False
            st.rerun()

# ====================== PERHITUNGAN ======================
if st.session_state.get("calculate", False):
    try:
        # Prepare data dari Excel
        df_clean = df_raw[['Year', 'gini_disp']].dropna().sort_values('Year').reset_index(drop=True)
        
        if len(df_clean) < 4:
            st.error("⚠️ Minimal **4 data** diperlukan untuk Double Exponential Smoothing.")
            st.stop()

        Y = df_clean['gini_disp'].values.astype(float)
        years = df_clean['Year'].values.astype(int)
        n = len(Y)

        # Double Exponential Smoothing (Brown) - satu parameter alpha
        S1 = [Y[0]]
        S2 = [Y[0]]
        for t in range(1, n):
            S1.append(alpha * Y[t] + (1 - alpha) * S1[t-1])
            S2.append(alpha * S1[t] + (1 - alpha) * S2[t-1])

        # Komponen a & b (Brown)
        a = [2 * S1[i] - S2[i] for i in range(n)]
        b = [((alpha / (1 - alpha)) * (S1[i] - S2[i])) if (1 - alpha) != 0 else 0.0 for i in range(n)]

        # Forecast in-sample (one-step ahead)
        forecast = [None]
        for i in range(1, n):
            forecast.append(a[i-1] + b[i-1])

        # Error calculation (mulai dari t=1)
        error = [None]
        abs_error = [None]
        error2 = [None]
        for i in range(1, n):
            f = forecast[i]
            if f is None:
                error.append(None)
                abs_error.append(None)
                error2.append(None)
            else:
                e = Y[i] - f
                error.append(e)
                abs_error.append(abs(e))
                error2.append(e**2)
        
        # Valid errors adalah yang bukan None
        valid_errors = [e for e in error if e is not None]
        valid_indices = [i for i in range(n) if error[i] is not None]
        
        if len(valid_errors) > 0:
            MAE = np.mean(np.abs(valid_errors))
            MSE = np.mean(np.square(valid_errors))
            RMSE = np.sqrt(MSE)
            # MAPE (gunakan abs_error dari i=1..n-1)
            valid_y = np.array([Y[i] for i in valid_indices if Y[i] != 0])
            valid_abs_error = np.array([abs_error[i] for i in valid_indices if Y[i] != 0])
            MAPE = float(np.nanmean(valid_abs_error / valid_y) * 100) if len(valid_y) > 0 else 0.0
        else:
            MAE = MSE = RMSE = MAPE = 0

        # Prediksi ke depan
        # Forecasting m steps ahead menggunakan a_n dan b_n
        future_years = [years[-1] + k + 1 for k in range(periods_ahead)]
        future_forecasts = [a[-1] + b[-1] * m for m in range(1, periods_ahead + 1)]
        future_years = [years[-1] + k + 1 for k in range(periods_ahead)]

        # ====================== HASIL ======================
        st.success(f"✅ Berhasil! Alpha = {alpha:.2f} | Data: {n} tahun | Prediksi {periods_ahead} tahun ke depan")

        # Tabel Perhitungan Lengkap
        st.subheader("📋 Tabel Perhitungan Lengkap")
        table_data = []
        for i in range(n):
            table_data.append({
                "No": i + 1,
                "Tahun": int(years[i]),
                "Gini Aktual": f"{Y[i]:.4f}",
                "S1": f"{S1[i]:.4f}",
                "S2": f"{S2[i]:.4f}",
                "a": f"{a[i]:.4f}",
                "b": f"{b[i]:.4f}",
                "Forecast": f"{forecast[i]:.4f}" if forecast[i] is not None else "-",
                "Error": f"{error[i]:.4f}" if error[i] is not None else "-",
                "|Error|": f"{abs_error[i]:.4f}" if abs_error[i] is not None else "-",
                "Error²": f"{error2[i]:.4f}" if error2[i] is not None else "-",
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # Prediksi Mendatang
        st.subheader(f"Prediksi {periods_ahead} Tahun ke Depan")
        pred_df = pd.DataFrame({
            "No": range(1, periods_ahead + 1),
            "Tahun": future_years,
            "Prediksi Gini": [f"{v:.4f}" for v in future_forecasts]
        })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

        # Metrik Evaluasi
        st.subheader("Metrik Evaluasi Model")
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"<div class='metric-card'><h3>{MAE:.4f}</h3><p>MAE</p><small>Mean Absolute Error</small></div>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"<div class='metric-card'><h3>{MSE:.4f}</h3><p>MSE</p><small>Mean Squared Error</small></div>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"<div class='metric-card'><h3>{RMSE:.4f}</h3><p>RMSE</p><small>Root Mean Squared Error</small></div>", unsafe_allow_html=True)
        with cols[3]:
            mape_color = "🟢" if MAPE < 5 else "🟡" if MAPE < 10 else "🟠" if MAPE < 20 else "🔴"
            mape_desc = "Sangat Baik" if MAPE < 5 else "Baik" if MAPE < 10 else "Cukup" if MAPE < 20 else "Perlu Perbaikan"
            st.markdown(f"<div class='metric-card'><h3>{mape_color} {MAPE:.2f}%</h3><p>MAPE</p><small>{mape_desc}</small></div>", unsafe_allow_html=True)

        # Grafik Visualisasi (matplotlib style seperti Colab)
        st.subheader("📈 Analisis Visual: Gini Aktual vs Forecast vs Prediksi")
        
        # Prepare data untuk visualisasi
        hasil_forecast_df = pd.DataFrame({
            'Year': list(years) + future_years,
            'Forecast_GINI_Disp': forecast + future_forecasts
        })
        
        # Buat figure matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual data
        ax.plot(years, Y, marker='o', label='Actual GINI_Disp', color='#00E396', linewidth=2, markersize=6)
        
        # Plot forecast (in-sample + future)
        ax.plot(hasil_forecast_df['Year'], hasil_forecast_df['Forecast_GINI_Disp'],
                marker='x', linestyle='--', label='Forecast GINI_Disp', color='#00D1FF', linewidth=2, markersize=8)
        
        ax.set_title('Forecasting GINI Dispersion (Double Exponential Smoothing)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('GINI Disp', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig)

        # ====================== PANDUAN ======================
        st.markdown("---")
        st.subheader("📘 Informasi Dataset & Metode")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='info-card'>
                <h5>Gini Coefficient</h5>
                <p><strong>Gini Coefficient</strong> adalah ukuran ketimpangan pendapatan. Semakin tinggi nilainya, semakin besar ketimpangan di suatu negara.</p>
                <p><small>Range: 0 (sempurna merata) hingga 1 (sempurna tidak merata)</small></p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class='info-card'>
                <h5>Pemilihan Alpha (α)</h5>
                <p>
                <strong>α → 0:</strong> Smoothing lambat, lebih stabil<br>
                <strong>α → 1:</strong> Sangat responsif terhadap data baru<br><br>
                <small><strong>Tips:</strong> Coba 0.1–0.3 untuk data stabil</small>
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class='info-card'>
                <h5>Interpretasi MAPE</h5>
                <p>
                <span style='background:#00E396;color:black;padding:4px 8px;border-radius:6px;font-size:0.9em;'><5%</span> → Sangat Baik<br>
                <span style='background:#00D1FF;color:black;padding:4px 8px;border-radius:6px;font-size:0.9em;'>5-10%</span> → Baik<br>
                <span style='background:#FEB019;color:black;padding:4px 8px;border-radius:6px;font-size:0.9em;'>10-20%</span> → Cukup<br>
                <span style='background:#EF4444;color:white;padding:4px 8px;border-radius:6px;font-size:0.9em;'>> 20%</span> → Perlu Perbaikan
                </p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
else:
    st.info("👈 Klik **Hitung Forecast** di sidebar untuk memulai analisis forecasting Gini Coefficient.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Tentang Dataset
        Dataset berisi data **Income Inequality in South Africa** dengan kolom:
        - **Year**: Tahun data
        - **gini_disps**: Gini Coefficient (disposable income)
        
        Metode **Double Exponential Smoothing** cocok untuk data dengan trend linier.
        """)
    with col2:
        st.markdown("""
        ### Cara Menggunakan
        1. Gunakan slider **Alpha (α)** untuk mengatur sensitivitas
        2. Pilih berapa tahun prediksi ke depan
        3. Klik **Hitung Forecast** untuk melihat hasil
        4. Gunakan kontrol grafik untuk menyesuaikan tampilan
        5. Lihat tabel dan metrik untuk analisis detail

        """)


