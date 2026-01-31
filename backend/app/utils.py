def likert_to_text(question: str, score: int) -> str:
    """
    Converts a numerical Likert score to a text description based on the trait.
    Match the format used in training data generation.
    """
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
        
    # If the question key is a trait
    if question in mapping:
        return mapping[question][idx]
    
    # Fallback for other questions (Standard Likert)
    scale = {
        1: "Strongly disagree",
        2: "Disagree",
        3: "Neutral",
        4: "Agree",
        5: "Strongly agree"
    }
    return f"{scale.get(score, 'Neutral')} with the statement: \"{question}\"."

def construct_prompt(general_scores: dict) -> str:
    """Constructs the input text for the model following the training data format."""
    profile_parts = []
    traits_order = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    
    for trait in traits_order:
        score = general_scores.get(trait, 3) 
        desc = likert_to_text(trait, score)
        profile_parts.append(desc)
        
    prompt = f"The user is described as: {', '.join(profile_parts[:-1])}, and {profile_parts[-1]}."
    return prompt
