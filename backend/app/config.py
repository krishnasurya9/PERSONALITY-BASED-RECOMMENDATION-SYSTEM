import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
BOOKS_API_KEY = os.getenv("BOOKS_API_KEY")
RAWG_API_KEY = os.getenv("RAWG_API_KEY")

# Model Path
# Using absolute path for safety, but now pointing to the new structure
MODEL_PATH = "A:/projects/project3/project3/models/best_bert_lstm.pth"

# Genre Mapping
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
