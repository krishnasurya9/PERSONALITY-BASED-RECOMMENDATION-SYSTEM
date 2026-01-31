from pydantic import BaseModel
from typing import Dict, List

class RecommendationRequest(BaseModel):
    domain: str
    general_scores: Dict[str, int]
    domain_responses: Dict[str, int]

class RecommendationResponse(BaseModel):
    dominant_trait: str
    personality_traits: Dict[str, float]
    recommended_genre: str
    recommendations: List[str]
    predicted_personality: str # Added for the updated frontend logic
    recommended_domain: str    # Added for the updated frontend logic
