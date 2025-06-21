import streamlit as st
import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.set_page_config(page_title="Book Genre Predictor", layout="centered")

st.markdown(
    "<h2 style='text-align: center;'>📚 Predict Your Most Preferred Book Genre</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Answer a few questions to find your ideal reading genre</p>",
    unsafe_allow_html=True
)

freq_options = encoders["Reading_Frequency"].classes_.tolist()
length_options = encoders["Book_Length"].classes_.tolist()
mood_options = encoders["Mood"].classes_.tolist()
genre_options = ['Fiction', 'Sci-Fi', 'Self-help', 'Biography', 'Thriller', 'Fantasy']

freq = st.selectbox("1️⃣ How often do you read books?", freq_options)
length = st.selectbox("2️⃣ What is your preferred book length?", length_options)
mood = st.selectbox("3️⃣ How do you usually feel while reading?", mood_options)
interested_genres = st.multiselect("4️⃣ Which genres do you enjoy reading?", genre_options)

if st.button("🎯 Predict Genre"):
    x_input = [
        encoders["Reading_Frequency"].transform([freq])[0],
        encoders["Book_Length"].transform([length])[0],
        encoders["Mood"].transform([mood])[0],
        int("Fiction" in interested_genres),
        int("Sci-Fi" in interested_genres),
        int("Self-help" in interested_genres),
        int("Biography" in interested_genres),
        int("Thriller" in interested_genres),
        int("Fantasy" in interested_genres)
    ]
    x_scaled = scaler.transform([x_input])
    pred = model.predict(x_scaled)[0]
    genre = encoders["Preferred_Genre"].inverse_transform([pred])[0]
    st.success(f"🎯 Your most preferred genre is likely: **{genre}**")

st.markdown(
    "<p style='text-align: center; font-size: 13px;'>Built with ❤️ for the ML Hackathon</p>",
    unsafe_allow_html=True
)
