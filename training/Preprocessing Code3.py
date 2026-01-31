import json
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "A:/projects/project3/data set/processed data/big_five_prompts.jsonl"

data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        data.append(entry)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Define label mapping (0, 1, 2 for each personality trait)
label_mapping = {
    "Very imaginative, loves new experiences": 2,
    "Somewhat open to new experiences": 1,
    "Prefers routine, less curious": 0,

    "Highly organized, goal-oriented": 2,
    "Moderately disciplined": 1,
    "Disorganized, spontaneous": 0,

    "Highly sociable, energetic": 2,
    "Sometimes outgoing, sometimes reserved": 1,
    "Introverted, prefers solitude": 0,

    "Very compassionate, trusts others easily": 2,
    "Moderately cooperative": 1,
    "Competitive, less empathetic": 0,

    "Prone to stress, emotionally reactive": 2,
    "Emotionally stable, calm": 0
}

# Convert personality descriptions into numerical labels
for trait in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]:
    df[trait] = df["response"].apply(lambda x: label_mapping.get(x[trait], -1))  # Use get() to avoid KeyErrors

# Tokenize text using BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(df["prompt"].tolist(), padding=True, truncation=True, return_tensors="pt")

# Split into training and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    inputs["input_ids"].tolist(),  # Convert tensor to list
    df[["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]].values,
    test_size=0.1, random_state=42
)

# Convert labels to PyTorch tensors
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Convert to DataFrame for saving
train_df = pd.DataFrame({
    "input_ids": train_texts,
    "Openness": train_labels[:, 0].tolist(),
    "Conscientiousness": train_labels[:, 1].tolist(),
    "Extraversion": train_labels[:, 2].tolist(),
    "Agreeableness": train_labels[:, 3].tolist(),
    "Neuroticism": train_labels[:, 4].tolist(),
})

test_df = pd.DataFrame({
    "input_ids": test_texts,
    "Openness": test_labels[:, 0].tolist(),
    "Conscientiousness": test_labels[:, 1].tolist(),
    "Extraversion": test_labels[:, 2].tolist(),
    "Agreeableness": test_labels[:, 3].tolist(),
    "Neuroticism": test_labels[:, 4].tolist(),
})

# Save DataFrames to CSV
train_df.to_csv("A:/projects/project3/PythonProject/train.csv", index=False)
test_df.to_csv("A:/projects/project3/PythonProject/test.csv", index=False)

print("Preprocessing complete!")
print(f"Training samples: {len(train_texts)}, Testing samples: {len(test_texts)}")
print("Train and test datasets saved successfully!")
