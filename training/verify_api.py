import os
import sys
# Force UTF-8 encoding for Windows consoles
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from fastapi.testclient import TestClient

# Add project root to sys.path so we can import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import app

client = TestClient(app)

def test_recommendation_endpoint():
    print("Testing /recommend endpoint...")
    
    payload = {
        "domain": "Books",
        "general_scores": {
            "Openness": 5,
            "Conscientiousness": 4,
            "Extraversion": 3,
            "Agreeableness": 4,
            "Neuroticism": 2
        },
        "domain_responses": {
            "Q1": 5,
            "Q2": 4,
            "Q3": 5
        }
    }
    
    try:
        response = client.post("/recommend", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("[OK] Request successful!")
            print(f"Dominant Trait: {data.get('dominant_trait')}")
            print(f"Recommended Genre: {data.get('recommended_genre')}")
            print("Traits Scores:")
            for trait, score in data.get('personality_traits', {}).items():
                print(f"  - {trait}: {score:.4f}")
            
            print("Recommendations:")
            for rec in data.get('recommendations', []):
                print(f"  - {rec}")
            
            # Basic validation
            assert "dominant_trait" in data, "Response missing dominant_trait"
            assert "personality_traits" in data, "Response missing personality_traits"
            assert len(data["personality_traits"]) == 5, "Should have 5 personality traits"
            assert "recommendations" in data, "Response missing recommendations"
            
            print("[OK] Response structure validated.")
        else:
            print(f"[FAIL] Request failed with status code: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"[ERROR] An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recommendation_endpoint()
