import streamlit as st
import pandas as pd

# Judul aplikasi
st.title("Tampilkan Kolom Dataset dalam Grafik")

# Input untuk mengunggah dataset
uploaded_file = st.file_uploader("car_evaluation_with.csv", type="csv")

if uploaded_file is not None:
    # Baca dataset
    df = pd.read_csv(car_evaluation_with.csv)
    
    # Tampilkan data frame
    st.write("Dataset Anda:")
    st.write(df)
    
    # Input multi select untuk memilih kolom
    selected_columns = st.multiselect("Pilih kolom yang ingin ditampilkan", df.columns)

    # Input memilih jenis grafik
    chart_type = st.selectbox("Pilih jenis grafik", ["Line Chart", "Bar Chart", "Area Chart"])

    # Tampilkan grafik berdasarkan pilihan kolom
    if selected_columns:
        for column in selected_columns:
            st.write(f"Grafik untuk kolom: {column}")
            if chart_type == "Line Chart":
                st.line_chart(df[column])
            elif chart_type == "Bar Chart":
                st.bar_chart(df[column])
            elif chart_type == "Area Chart":
                st.area_chart(df[column])
    else:
        st.write("Pilih minimal satu kolom untuk menampilkan grafik.")
else:
    st.write("Silakan unggah dataset terlebih dahulu.")
