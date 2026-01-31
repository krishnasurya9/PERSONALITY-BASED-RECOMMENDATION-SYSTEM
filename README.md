# Personal Taste Engine

A Full-Stack AI Application that provides entertainment recommendations (Movies, Books, Music, Games) based on your personality traits.

## 🌟 Features
- **Personality Analysis**: Uses a fine-tuned BERT-LSTM model to analyze your Big Five personality traits from text responses.
- **Dynamic Recommendations**: Recommends varied genres based on your specific personality profile.
- **Multi-Domain**: Covers Movies, Books, Music, and Games.
- **Modern UI**: A sleek, dark-themed interface with interactive elements and animations.

## 📁 Project Structure
```
project3/
├── backend/            # FastAPI Backend
│   └── app/
│       ├── main.py     # API Entry Point
│       ├── model.py    # AI Model Definition
│       ├── services.py # External API Integration
│       └── ...
├── frontend/           # Web Interface
├── data/               # Training Datasets
└── models/             # Trained PyTorch Model
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- API Keys for TMDB, Google Books, YouTube, and RAWG (stored in `.env`).

### 1. Backend Setup
Navigate to the backend directory:
```bash
cd backend
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file in `backend/` with your keys:
```env
TMDB_API_KEY=your_key
YOUTUBE_API_KEY=your_key
BOOKS_API_KEY=your_key
RAWG_API_KEY=your_key
```

Run the server:
```bash
python -m app.main
```
The API will run at `http://127.0.0.1:8000`.

### 2. Frontend Setup
You can simply open `frontend/index.html` in your browser.
For the best experience, run a local server:
```bash
cd ../frontend
python -m http.server 8080
```
Open `http://127.0.0.1:8080`.

## 🧠 Model Details
The system uses a **BERT-LSTM** architecture fine-tuned on the Big Five personality dataset. It takes natural language responses as input and predicts a 5-dimensional personality vector (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism).

## 📝 Credits
Developed as part of Project 3.
