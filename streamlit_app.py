import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load models
@st.cache_resource
def load_models():
    lstm_model = load_model('lstm_model.h5')
    knn_model = joblib.load('knn_model.pkl')
    return lstm_model, knn_model

lstm_model, knn_model = load_models()

# Fungsi untuk prediksi
def predict(input_data, lstm_model, knn_model):
    # Scale input data
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Reshape untuk input LSTM (samples, timesteps, features)
    input_data_lstm = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))
    
    # Ekstraksi fitur menggunakan model LSTM
    features = lstm_model.predict(input_data_lstm)
    
    # Pastikan features memiliki 4 fitur sesuai dengan KNN
    if features.shape[1] != 4:
        raise ValueError(f"Output LSTM hanya memiliki {features.shape[1]} fitur, tapi KNN memerlukan 4 fitur.")
    
    # Prediksi hasil menggunakan model KNN
    prediction = knn_model.predict(features)
    return prediction

# Interface Streamlit
st.title("Car Evaluation Prediction App")
st.write("Aplikasi ini memprediksi hasil evaluasi mobil berdasarkan input Anda.")

# Input pengguna
input_data = []
input_data.append(st.number_input("Buying (0-3)", min_value=0, max_value=3, step=1))
input_data.append(st.number_input("Maint (0-3)", min_value=0, max_value=3, step=1))
input_data.append(st.number_input("Doors (2-5)", min_value=2, max_value=5, step=1))
input_data.append(st.number_input("Persons (2-5)", min_value=2, max_value=5, step=1))
input_data.append(st.number_input("Lug_boot (0-2)", min_value=0, max_value=2, step=1))
input_data.append(st.number_input("Safety (0-2)", min_value=0, max_value=2, step=1))

# Ubah input data menjadi bentuk array
input_data = np.array(input_data).reshape(1, -1)

# Prediksi
if st.button("Predict"):
    try:
        result = predict(input_data, lstm_model, knn_model)
        st.success(f"Hasil prediksi evaluasi mobil adalah: {result[0]}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
