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
│       └── ...
├── frontend/           # Web Interface
├── data/               # Training Datasets (CSVs/JSONLs excluded from repo)
├── models/             # Trained PyTorch Model (Binaries excluded from repo)
├── training/           # Model Training Scripts
│   ├── fine tunning.py # Main training script
│   └── evaluation.py   # Evaluation scripts
└── evaluation/         # Performance Plots & Results
    ├── plots/          # Confusion Matrices, ROC Curves
    └── results.txt     # Numeric metrics
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- API Keys for TMDB, Google Books, YouTube, and RAWG (stored in `.env`).

### 1. Backend Setup
Navigate to the backend directory:
```bash
cd backend
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
cd frontend
python -m http.server 8080
```
Open `http://127.0.0.1:8080`.

## 🧠 Model Training (Required)

**Note:** The trained model files (`.pth`) and large datasets are **not included** in this repository to keep it lightweight. You must train the model locally to use the personality analysis feature.

### How to Train the Model
1.  **Prepare Data**: Place your Big Five dataset (JSONL format) in `data/processed data/big_five_prompts.jsonl`.
2.  **Run Training Script**:
    ```bash
    cd training
    python "fine tunning.py"
    ```
3.  **Output**:
    - The script will train the BERT-LSTM model on your GPU (if available) or CPU.
    - The best model will be saved to `models/best_bert_lstm.pth`.
    - Logs are saved to `status.txt`.

## 📊 Evaluation Results

We have included comprehensive evaluation metrics for the model in the `evaluation/` directory.

- **Plots**: View confusion matrices and ROC curves in `evaluation/plots/`.
- **Metrics**: Detailed accuracy, precision, recall, and F1-scores are available in `evaluation/results.txt`.

### Model Architecture
The system uses a **BERT-LSTM** architecture:
1.  **BERT (Base Uncased)**: Extracts contextual embeddings from user responses.
2.  **Bi-Directional LSTM**: Captures sequential dependencies in the text.
3.  **Fully Connected Layer**: Maps the output to 5 personality trait scores.

## 📝 Credits
Developed as part of Project 3.
