import requests
from typing import List
from .config import TMDB_API_KEY, BOOKS_API_KEY, YOUTUBE_API_KEY, RAWG_API_KEY

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
        # Fallback recommendations if API fails
        return [f"Recommendation 1 for {genre}",
                f"Recommendation 2 for {genre}",
                f"Recommendation 3 for {genre}"]


def fetch_movie_recommendations(genre: str) -> List[str]:
    """Fetch movie recommendations from TMDB API."""
    tmdb_genre_ids = {
        "Comedy": 35,
        "Drama": 18,
        "Action": 28,
        "Fantasy": 14,
        "Historical Fiction": 36, # Using History
        "Mystery": 9648
    }

    # TMDB Genre Mapping improvements
    if genre == "Historical Fiction": genre_id = 36
    elif genre == "Fantasy": genre_id = 14
    else: genre_id = tmdb_genre_ids.get(genre, 0)
    
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
        return [book["volumeInfo"].get("title", "Unknown Title") for book in books]
    except Exception as e:
        print(f"Google Books API error: {str(e)}")
        return [f"{genre} Book 1", f"{genre} Book 2", f"{genre} Book 3"]


def fetch_music_recommendations(genre: str) -> List[str]:
    """Fetch music recommendations from YouTube API."""
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
