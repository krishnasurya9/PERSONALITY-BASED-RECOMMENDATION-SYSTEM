# PERSONALITY-BASED-RECOMMENDATION-SYSTEM
This Personality-Based Recommendation System leverages Big Five personality traits. It predicts user personality via custom questions and a BERT-LSTM model, then generates personalized recommendations across entertainment domains like movies and music. The goal is to enhance personalization through deeper psychological insights

# Personality-Informed Recommendation System

A full-stack AI system that predicts a user's personality from survey responses and recommends personalized content (movies, books, music, or games) using a trained deep learning model.

---

## 🔧 Tech Stack
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, FastAPI
- **ML/NLP:** BERT, LSTM, Transformers, Scikit-learn
- **Other:** Pandas, NumPy, Matplotlib, Streamlit

---

## 📁 Project Structure

```
Personality_Recommender_Full_Stack/
│
├── FRONTEND/                # Frontend files
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── main.py                  # FastAPI app
├── fine tunning.py          # Model fine-tuning
├── evaluation.py            # Evaluation script
├── inference_cli.py         # Inference utility
├── preprocess*.py           # Data preprocessing
├── requirements.txt         # Dependencies
there are some files which are not menction they are Additional files which are used for the preprocessing
```

---

## ▶️ How to Run the Project

### 1. Backend (FastAPI)
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2. Frontend
Open `FRONTEND/index.html` in your browser.

---

## 💡 Features

- Predicts Big Five personality traits from survey data
- Generates personalized content recommendations
- Clean UI with interactive inputs
- Modular codebase for easy extension

---

## 📌 Notes

- All code runs locally. No external server or deployment is needed.
- For demo purposes, include screenshots or video recordings of the UI interaction.
NOTE--- THERE ARE NO API KEYS IN CODE IF YOU NEED ADD YOUR KEYS HERE ARE THE WEBSITES FOR THE KEYS 
Games APIs
RAWG Video Games Database API
🔗 https://rawg.io/apidocs
Google Books API
🔗 https://console.cloud.google.com/apis/library/books.googleapis.com
TMDB (The Movie Database)
🔗 https://www.themoviedb.org/documentation/api
For Music APIs i used same as the google books i used youtube api which is linked in with google books 
