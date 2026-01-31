print(">>> CLEAN main.py is running")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import requests
import statistics
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Entertainment Recommendation API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model definitions
class RecommendationRequest(BaseModel):
    domain: str
    general_scores: Dict[str, int]
    domain_responses: Dict[str, int]


class RecommendationResponse(BaseModel):
    dominant_trait: str
    personality_traits: Dict[str, float]
    recommended_genre: str
    recommendations: List[str]


# API keys for external services (Loaded from .env)
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
BOOKS_API_KEY = os.getenv("BOOKS_API_KEY")
RAWG_API_KEY = os.getenv("RAWG_API_KEY")

# Genre mapping based on personality and domain responses
GENRE_MAPPING = {
    "Books": {
        "high": "Fantasy",
        "low": "Historical Fiction",
        "mid": "Mystery"
    },
    "Movies": {
        "high": "Drama",
        "low": "Comedy",
        "mid": "Action"
    },
    "Music": {
        "high": "Classical",
        "low": "Pop",
        "mid": "Rock"
    },
    "Games": {
        "high": "RPG",
        "low": "Arcade",
        "mid": "Strategy"
    }
}

# === Model Architecture (Must match training and inference_cli) ===
class BERT_LSTM(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', lstm_hidden_dim=256, output_dim=5):
        super(BERT_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)  # 256*2=512

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        _, (hidden, _) = self.lstm(sequence_output)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # because bidirectional=True
        return self.fc(hidden)

# === Global Model and Tokenizer Loading ===
MODEL_PATH = "A:/projects/project3/models/best_bert_lstm.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None
model = None

try:
    print("... Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("... Loading model...")
    model = BERT_LSTM()
    # Load state dict with map_location to handle CPU/GPU differences
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("[OK] Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    # We don't crash the app, but inference will fail if called


# === Helper Functions defined in inference_cli.py ===

personality_questions = {
    "Openness": "Do you enjoy trying new things?",
    "Conscientiousness": "Are you a detail-oriented person?",
    "Extraversion": "Do you gain energy from social interactions?",
    "Agreeableness": "Do you prioritize harmony in relationships?",
    "Neuroticism": "Do you often feel anxious or stressed?"
}

domain_questions = {
    "Books": {
        "Q1": "Do you enjoy books with magical elements?",
        "Q2": "Do you prefer fiction over non-fiction?",
        "Q3": "Do you like shorter or longer books?"
    },
    "Movies": {
        "Q1": "Do you enjoy emotional dramas?",
        "Q2": "Do you prefer action over comedy?",
        "Q3": "Do you like watching thrillers?"
    },
    "Music": {
        "Q1": "Do you enjoy relaxing music more than energetic?",
        "Q2": "Do you prefer instrumental over lyrical?",
        "Q3": "Do you often explore new genres?"
    },
    "Games": {
        "Q1": "Do you enjoy story-driven games?",
        "Q2": "Do you prefer multiplayer over single-player?",
        "Q3": "Do you enjoy open-world exploration?"
    }
}

def likert_to_text(question, score):
    mapping = {
        "Openness": ["Prefers routine, less curious", "Somewhat open to new experiences",
              "Very imaginative, loves new experiences"],
        "Conscientiousness": ["Disorganized, spontaneous", "Moderately disciplined", "Highly organized, goal-oriented"],
        "Extraversion": ["Introverted, prefers solitude", "Sometimes outgoing, sometimes reserved", "Highly sociable, energetic"],
        "Agreeableness": ["Competitive, less empathetic", "Moderately cooperative", "Very compassionate, trusts others easily"],
        "Neuroticism": ["Emotionally stable, calm", "Occasionally stressed, manages emotions",
              "Prone to stress, emotionally reactive"]
    }
    
    score = int(score)
    # Mapping 1-5 scale to 0-2 indices
    # 1-2 -> Index 0 (Low)
    # 3   -> Index 1 (Medium)
    # 4-5 -> Index 2 (High)
    
    if score <= 2:
        idx = 0
    elif score == 3:
        idx = 1
    else:
        idx = 2
        
    # If the question key isn't a trait (e.g. domain questions), fallback to old logic or skip
    if question in mapping:
        return mapping[question][idx]
    
    # Fallback for domain questions (keep old logic or similar)
    scale = {
        1: "Strongly disagree",
        2: "Disagree",
        3: "Neutral",
        4: "Agree",
        5: "Strongly agree"
    }
    return f"{scale.get(score, 'Neutral')} with the statement: \"{question}\"."


@app.get("/")
def read_root():
    return {"message": "Entertainment Recommendation API"}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    try:
        # === 1. Construct Input Text for Model ===
        # Match the format used in inference_cli.py gather_inputs()
        
        # General personality text construction (Training Data Format)
        # "The user is described as: [Desc1], [Desc2], ..."
        profile_parts = []
        traits_order = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        
        for trait in traits_order:
            score = request.general_scores.get(trait, 3) # Default to 3
            # Pass usage of trait name as key for mapping
            desc = likert_to_text(trait, score)
            profile_parts.append(desc)
            
        prompt_intro = f"The user is described as: {', '.join(profile_parts[:-1])}, and {profile_parts[-1]}."
        
        # Domain specific text - append as before? 
        # The training data DOES NOT appear to have domain text appended (based on the sample).
        # Adding domain text might confuse the model if it wasn't trained on it.
        # However, inference_cli.py appended it. 
        # Given the training data strictly follows the "The user is described as: ..." pattern (Step 111),
        # appending domain text ("Strongly agree with...") might add noise.
        # But the USER's inference_cli.py did it.
        # HYPOTHESIS: The user intended to use domain text for *something* but maybe the model ignores it?
        # OR the model in use (BERT) can handle extra tokens.
        # SAFE BET: Keep domain text but separating it clearly, OR strictly follow training data.
        # User said "I didn't connect at all".
        # If I want "good answers", I should match training data exactly.
        # The domain questions seem to be used for RULE-BASED logic (genre mapping) in main.py, 
        # while the MODEL predicts traits from interactions?
        # actually, the model predicts traits from TEXT.
        # If we already HAVE the score (request.general_scores), why predict it?
        # Ah, the architecture is: User Answers -> Text -> Model -> Predicted Traits -> Recommendation.
        # This seems redundant (Scores -> Text -> Model -> Scores).
        # BUT this is what the user built. The "Model" is likely a "Refiner" or "Personality Detector".
        # Let's produce the text exactly as the training data expects.
        
        full_text = prompt_intro
        
        # === 2. Run Inference if Model Loaded ===
        personality_traits = {}
        dominant_trait = "Unknown"
        
        if model and tokenizer:
            encoded = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in encoded.items()}
            
            with torch.no_grad():
                output = model(inputs["input_ids"], inputs["attention_mask"])
            
            scores = torch.sigmoid(output).squeeze().tolist()
            traits_list = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
            personality_traits = dict(zip(traits_list, scores))
            
            # Determine dominant trait
            dominant_trait = max(personality_traits.items(), key=lambda x: x[1])[0]
        else:
            # Fallback if model failed to load
            print("[WARN] Using fallback heuristic (Model not loaded)")
            personality_traits = {trait: float(score)/5.0 for trait, score in request.general_scores.items()}
            dominant_trait = max(personality_traits.items(), key=lambda x: x[1])[0]


        # === 3. Determine Genre (Heuristic or Logic) ===
        # Calculate average score across all responses (Heuristic from original code)
        all_scores = list(request.general_scores.values()) + list(request.domain_responses.values())
        avg_score = statistics.mean(all_scores)

        # Determine genre based on average score (Legacy logic, maybe improve later?)
        if avg_score >= 4:
            score_category = "high"
        elif avg_score <= 2:
            score_category = "low"
        else:
            score_category = "mid"

        recommended_genre = GENRE_MAPPING.get(request.domain, {}).get(score_category, "General")

        # === 4. Fetch Recommendations ===
        recommendations = fetch_recommendations(request.domain, recommended_genre)

        return RecommendationResponse(
            dominant_trait=dominant_trait,
            personality_traits=personality_traits,
            recommended_genre=recommended_genre,
            recommendations=recommendations
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


def fetch_recommendations(domain: str, genre: str) -> List[str]:
    """Fetch recommendations from external APIs based on domain and genre."""
    try:
        if domain == "Movies":
            return fetch_movie_recommendations(genre)
        elif domain == "Books":
            return fetch_book_recommendations(genre)
        elif domain == "Music":
            return fetch_music_recommendations(genre)
        elif domain == "Games":
            return fetch_game_recommendations(genre)
        else:
            return ["No recommendations available for this domain"]
    except Exception as e:
        print(f"Error fetching recommendations: {str(e)}")
        return [f"Recommendation 1 for {genre}",
                f"Recommendation 2 for {genre}",
                f"Recommendation 3 for {genre}"]


def fetch_movie_recommendations(genre: str) -> List[str]:
    """Fetch movie recommendations from TMDB API."""
    tmdb_genre_ids = {
        "Comedy": 35,
        "Drama": 18,
        "Action": 28
    }

    genre_id = tmdb_genre_ids.get(genre, 0)
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {"api_key": TMDB_API_KEY, "sort_by": "popularity.desc"}

    if genre_id:
        params["with_genres"] = genre_id

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        movies = response.json().get("results", [])
        return [movie["title"] for movie in movies[:5]]
    except Exception as e:
        print(f"TMDB API error: {str(e)}")
        return [f"{genre} Movie 1", f"{genre} Movie 2", f"{genre} Movie 3"]


def fetch_book_recommendations(genre: str) -> List[str]:
    """Fetch book recommendations from Google Books API."""
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": f"subject:{genre}", "key": BOOKS_API_KEY, "maxResults": 5}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        books = response.json().get("items", [])
        return [book["volumeInfo"]["title"] for book in books]
    except Exception as e:
        print(f"Google Books API error: {str(e)}")
        return [f"{genre} Book 1", f"{genre} Book 2", f"{genre} Book 3"]


def fetch_music_recommendations(genre: str) -> List[str]:
    """Fetch music recommendations from YouTube API."""
    # YouTube Search API is expensive/limited, but following original pattern
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": f"{genre} music",
        "type": "video",
        "videoCategoryId": "10",
        "maxResults": 5,
        "key": YOUTUBE_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        videos = response.json().get("items", [])
        return [video["snippet"]["title"] for video in videos]
    except Exception as e:
        print(f"YouTube API error: {str(e)}")
        return [f"{genre} Song 1", f"{genre} Song 2", f"{genre} Song 3"]


def fetch_game_recommendations(genre: str) -> List[str]:
    """Fetch game recommendations from RAWG API."""
    url = "https://api.rawg.io/api/games"
    params = {"key": RAWG_API_KEY, "genres": genre.lower(), "page_size": 5}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        games = response.json().get("results", [])
        return [game["name"] for game in games]
    except Exception as e:
        print(f"RAWG API error: {str(e)}")
        return [f"{genre} Game 1", f"{genre} Game 2", f"{genre} Game 3"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
