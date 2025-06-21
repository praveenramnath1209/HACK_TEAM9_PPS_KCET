import streamlit as st
import pickle
import numpy as np

# Load model, scaler, encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Dynamically get valid dropdown options from encoders
freq_options = [opt.encode('latin1').decode('utf-8') for opt in encoders['Reading_Frequency'].classes_.tolist()]
length_options = [opt.encode('latin1').decode('utf-8') for opt in encoders['Book_Length'].classes_.tolist()]
mood_options = [opt.encode('latin1').decode('utf-8') for opt in encoders['Mood'].classes_.tolist()]

# Genre checkbox options
genres = ['Fiction', 'Sci-Fi', 'Self-help', 'Biography', 'Thriller', 'Fantasy']

def main():
    st.set_page_config(page_title="📚 Book Genre Predictor", page_icon="📖", layout="centered")

    st.markdown("""
        <style>
        .title {
            text-align: center;
            color: #ffffff;
        }
        .footer {
            text-align: center;
            font-size: 12px;
            color: #aaaaaa;
            padding-top: 30px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h2 class='title'>📚 Predict Your Most Preferred Book Genre</h2>
    <p style='text-align:center;'>Answer a few questions to find your ideal reading genre</p>
    """, unsafe_allow_html=True)

    with st.form("genre_form"):
        freq = st.selectbox("1️⃣ How often do you read books?", freq_options)
        length = st.selectbox("2️⃣ What is your preferred book length?", length_options)
        mood = st.selectbox("3️⃣ How do you usually feel while reading?", mood_options)
        selected_genres = st.multiselect("4️⃣ Which genres do you enjoy reading?", genres)

        submitted = st.form_submit_button("🔍 Predict Genre")

    if submitted:
        try:
            # Re-encode user inputs
            freq_enc = encoders['Reading_Frequency'].transform([freq])[0]
            length_enc = encoders['Book_Length'].transform([length])[0]
            mood_enc = encoders['Mood'].transform([mood])[0]

            genre_flags = [1 if g in selected_genres else 0 for g in genres]
            input_vector = np.array([[freq_enc, length_enc, mood_enc] + genre_flags])
            input_scaled = scaler.transform(input_vector)

            prediction = model.predict(input_scaled)[0]
            predicted_genre = encoders['Preferred_Genre'].inverse_transform([prediction])[0]

            st.success(f"🎯 Your most preferred genre is likely: **{predicted_genre}**")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")

    st.markdown("""
    <div class='footer'>
        Built with ❤️ for the ML Hackathon
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
