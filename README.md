# 📚 Book Genre Prediction – ML Hackathon Project

This project predicts a user’s preferred book genre based on their reading preferences using semi-supervised machine learning.

---

## 📌 Problem Statement

> Predict the most likely book genre a reader would prefer based on their answers to a form (reading frequency, mood, author, etc.). Only a portion of data is labeled.

---

## 🔧 Technologies Used

- Python 3
- Google Colab
- Pandas, Numpy
- Scikit-learn
- Streamlit (for UI deployment)
- Matplotlib, Seaborn (optional for visualization)

---

## 🧠 Model

We use `SelfTrainingClassifier` (semi-supervised) from `scikit-learn` on partially labeled form data to train a genre prediction model.

---

## 🚀 How to Run

1. Upload `responses.csv` from Google Forms
2. Train model in Colab using `ml_code.ipynb`
3. Save the model as `model.pkl`
4. Use `app.py` and `requirements.txt` to deploy the Streamlit app
5. Push to GitHub and deploy via [https://streamlit.io/cloud](https://streamlit.io/cloud)

---

## 📁 Repo Structure

