import pandas as pd
import json

# Load the processed Big Five dataset
file_path = "\\projects\\project3\\data set\\big five\\cleaned_data.csv"  # Update if needed
df = pd.read_csv(file_path)


# Define mapping function
def map_trait_score(trait, score):
    mapping = {
        "O": ["Prefers routine, less curious", "Somewhat open to new experiences",
              "Very imaginative, loves new experiences"],
        "C": ["Disorganized, spontaneous", "Moderately disciplined", "Highly organized, goal-oriented"],
        "E": ["Introverted, prefers solitude", "Sometimes outgoing, sometimes reserved", "Highly sociable, energetic"],
        "A": ["Competitive, less empathetic", "Moderately cooperative", "Very compassionate, trusts others easily"],
        "N": ["Emotionally stable, calm", "Occasionally stressed, manages emotions",
              "Prone to stress, emotionally reactive"]
    }

    if score <= 2:
        return mapping[trait][0]
    elif 3 <= score <= 5:
        return mapping[trait][1]
    else:
        return mapping[trait][2]


# Convert numerical scores to text descriptions
converted_data = []
for _, row in df.iterrows():
    profile = {
        "Openness": map_trait_score("O", row["Openness"]),
        "Conscientiousness": map_trait_score("C", row["Conscientiousness"]),
        "Extraversion": map_trait_score("E", row["Extraversion"]),
        "Agreeableness": map_trait_score("A", row["Agreeableness"]),
        "Neuroticism": map_trait_score("N", row["Neuroticism"]),
    }
    prompt = f"The user is described as: {profile['Openness']}, {profile['Conscientiousness']}, {profile['Extraversion']}, {profile['Agreeableness']}, and {profile['Neuroticism']}."
    converted_data.append({"prompt": prompt, "response": profile})

# Save to JSONL format for LLM fine-tuning
output_path = "big_five_prompts.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for entry in converted_data:
        f.write(json.dumps(entry) + "\n")

print(f"Converted data saved to {output_path}")
