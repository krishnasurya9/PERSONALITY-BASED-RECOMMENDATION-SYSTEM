import uuid
import requests

# === Personality Questions ===
personality_questions = {
    "Openness": "Do you enjoy trying new things?",
    "Conscientiousness": "Are you a detail-oriented person?",
    "Extraversion": "Do you gain energy from social interactions?",
    "Agreeableness": "Do you prioritize harmony in relationships?",
    "Neuroticism": "Do you often feel anxious or stressed?"
}

domain_specific_questions = {
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

def ask_likert(question):
    while True:
        try:
            score = int(input(f"{question} (1-5): "))
            if 1 <= score <= 5:
                return score
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def ask_general_personality():
    return {trait: ask_likert(q) for trait, q in personality_questions.items()}

def ask_domain_selection():
    while True:
        choice = input("Enter a domain (Books, Movies, Music, Games): ").capitalize()
        if choice in domain_specific_questions:
            return choice
        else:
            print("Invalid domain. Try again.")

def ask_domain_questions(domain):
    questions = domain_specific_questions[domain]
    return {k: ask_likert(v) for k, v in questions.items()}

def calculate_recommendation(domain, general_scores, domain_scores):
    all_scores = list(general_scores.values()) + list(domain_scores.values())
    avg = sum(all_scores) / len(all_scores)

    if domain == "Books":
        return "Fantasy" if avg >= 3.5 else "Historical"
    elif domain == "Movies":
        return "Drama" if avg >= 3.5 else "Comedy"
    elif domain == "Music":
        return "Classical" if avg >= 3.5 else "Pop"
    elif domain == "Games":
        return "RPG" if avg >= 3.5 else "Arcade"
    return "General"

# === Real-Time APIs ===

TMDB_API_KEY = ""
def fetch_movie_recommendations(genre):
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
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        return [m["title"] for m in res.json().get("results", [])[:5]]
    except Exception as e:
        print("TMDb error:", e)
        return []

YOUTUBE_API_KEY = ""
def fetch_music_recommendations(genre):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": genre + " music",
        "type": "video",
        "videoCategoryId": "10",
        "maxResults": 5,
        "key": YOUTUBE_API_KEY
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        return [f"{item['snippet']['title']} - https://www.youtube.com/watch?v={item['id']['videoId']}" for item in data.get("items", [])]
    except Exception as e:
        print("YouTube API error:", e)
        return []

BOOKS_API_KEY = "
def fetch_book_recommendations(genre):
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": f"subject:{genre}", "key": BOOKS_API_KEY, "maxResults": 5}
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        return [book["volumeInfo"]["title"] for book in res.json().get("items", [])]
    except Exception as e:
        print("Google Books API error:", e)
        return []

RAWG_API_KEY = ""
def fetch_game_recommendations(genre):
    url = "https://api.rawg.io/api/games"
    params = {"key": RAWG_API_KEY, "genres": genre.lower(), "page_size": 5}
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        return [g["name"] for g in res.json().get("results", [])]
    except Exception as e:
        print("RAWG API error:", e)
        return []

def main():
    user_id = str(uuid.uuid4())
    print(f"\n🎯 Your Session ID: {user_id}")

    print("\n🧠 General Personality Questions")
    general_scores = ask_general_personality()

    print("\n🎭 Entertainment Domain Selection")
    domain = ask_domain_selection()

    print(f"\n📋 {domain} Domain Questions")
    domain_scores = ask_domain_questions(domain)

    genre = calculate_recommendation(domain, general_scores, domain_scores)
    print(f"\n✅ Based on your answers, recommended genre: {genre}")

    print(f"\n🌐 Fetching real-time recommendations for {domain}...\n")
    if domain == "Books":
        recs = fetch_book_recommendations(genre)
    elif domain == "Movies":
        recs = fetch_movie_recommendations(genre)
    elif domain == "Music":
        recs = fetch_music_recommendations(genre)
    elif domain == "Games":
        recs = fetch_game_recommendations(genre)
    else:
        recs = []

    if recs:
        for r in recs:
            print(" -", r)
    else:
        print("No real-time results found.")

if __name__ == "__main__":
    main()
