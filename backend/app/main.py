from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import statistics

# Import from local modules
from .config import GENRE_MAPPING
from .schemas import RecommendationRequest, RecommendationResponse
from .services import fetch_recommendations
from .model import load_ai_model, get_model
from .utils import construct_prompt

# Application Setup
app = FastAPI(title="Personal Taste Engine API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model on Startup
@app.on_event("startup")
async def startup_event():
    load_ai_model()

@app.get("/")
def read_root():
    return {"message": "Entertainment Recommendation API is running."}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    try:
        # 1. Construct Input Prompt
        full_text = construct_prompt(request.general_scores)
        
        # 2. Run Inference
        model, tokenizer, device = get_model()
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
            
            dominant_trait = max(personality_traits.items(), key=lambda x: x[1])[0]
        else:
            # Fallback
            print("[WARN] Model not loaded, using fallback.")
            personality_traits = {trait: float(score)/5.0 for trait, score in request.general_scores.items()}
            dominant_trait = max(personality_traits.items(), key=lambda x: x[1])[0]

        # 3. Determine Genre (Heuristic)
        all_scores = list(request.general_scores.values()) + list(request.domain_responses.values())
        avg_score = statistics.mean(all_scores)

        if avg_score >= 4: score_category = "high"
        elif avg_score <= 2: score_category = "low"
        else: score_category = "mid"

        recommended_genre = GENRE_MAPPING.get(request.domain, {}).get(score_category, "General")

        # 4. Fetch Recommendations
        recommendations = fetch_recommendations(request.domain, recommended_genre)

        return RecommendationResponse(
            dominant_trait=dominant_trait,
            personality_traits=personality_traits,
            recommended_genre=recommended_genre,
            recommendations=recommendations,
            predicted_personality=f"{dominant_trait} Personality",
            recommended_domain=request.domain
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# For debugging/running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
