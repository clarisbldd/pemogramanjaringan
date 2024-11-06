import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load models
lstm_model = load_model('lstm_model.h5')
with open('knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

# Fungsi untuk prediksi
def predict(input_data):
    # Scale input data
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Reshape untuk input LSTM (samples, timesteps, features)
    input_data_lstm = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))

    # Ekstraksi fitur menggunakan model LSTM
    features = lstm_model.predict(input_data_lstm)

    # Prediksi hasil menggunakan model KNN
    prediction = knn_model.predict(features)
    return prediction

# Interface Streamlit
st.title("Car Evaluation Prediction App")
st.write("Aplikasi ini memprediksi hasil evaluasi mobil berdasarkan input Anda.")

# Input pengguna
# Untuk contoh, kita anggap dataset memiliki 6 fitur yang pengguna perlu input.
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
    result = predict(input_data)
    st.write(f"Hasil prediksi evaluasi mobil adalah: {result[0]}")
