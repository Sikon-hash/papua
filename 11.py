import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "Visualisasi", "Prediksi Curah Hujan", "Upload Dataset"])

if page == "Beranda":
    st.image("money.jpg")
    st.title("Aplikasi Analisis Curah Hujan Papua Barat")
    st.write("Selamat datang! Aplikasi ini menampilkan visualisasi data curah hujan dan prediksi menggunakan model machine learning.")

@st.cache_data
def load_data():
    df = pd.read_csv("curah_hujan_papua.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

if page == "Visualisasi":
    st.subheader("Dataset Curah Hujan")
    st.dataframe(df)

    if 'Time' in df.columns and 'precipitation' in df.columns:
        st.subheader("Visualisasi Bar Chart: Time vs Precipitation")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df['Time'], df['precipitation'], color='skyblue')
        ax.set_xlabel("Time")
        ax.set_ylabel("Precipitation")
        ax.set_title("Time vs Precipitation")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Visualisasi Line Chart")
        st.line_chart(df.set_index('Time')['precipitation'])
    else:
        st.warning("Kolom 'Time' dan/atau 'precipitation' tidak ditemukan.")

if page == "Prediksi Curah Hujan":
    st.subheader("Model Regresi Linear: Prediksi Curah Hujan")

    if 'precipitation' in df.columns and df.select_dtypes(include=['number']).shape[1] > 1:
        X = df.select_dtypes(include=['number']).drop(columns=['precipitation'])
        y = df['precipitation']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.success(f"Model dilatih dengan MSE: {mse:.2f}")
        st.write("Masukkan nilai input untuk prediksi:")
        input_data = {}
        for col in X.columns:
            val = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()))
            input_data[col] = val

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.info(f"Prediksi Curah Hujan: **{prediction:.2f} mm**")
    else:
        st.warning("Dataset tidak memiliki fitur numerik cukup untuk prediksi.")

if page == "Upload Dataset":
    st.subheader("Upload Dataset CSV Baru")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        st.write("Dataset yang Diupload:")
        st.dataframe(new_df)
