
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

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




# Load the saved model
model_path = "A:/projects/project3/models/best_bert_lstm.pth"  # Replace with your path
model = BERT_LSTM()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# === Questions ===

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

def ask_likert(question):
    while True:
        try:
            value = int(input(f"{question} (1-5): "))
            if value in [1,2,3,4,5]:
                return value
            else:
                print("Enter a value between 1 and 5.")
        except:
            print("Enter a number.")

def likert_to_text(question, score):
    scale = {
        1: "Strongly disagree",
        2: "Disagree",
        3: "Neutral",
        4: "Agree",
        5: "Strongly agree"
    }
    return f"{scale[score]} with the statement: \"{question}\"."

def gather_inputs():
    print("\n--- General Personality Questions ---")
    general_scores = {trait: ask_likert(q) for trait, q in personality_questions.items()}

    print("\nChoose an entertainment domain: Books / Movies / Music / Games")
    while True:
        domain = input("Your choice: ").capitalize()
        if domain in domain_questions:
            break
        else:
            print("Invalid domain. Try again.")

    print(f"\n--- {domain} Domain Questions ---")
    domain_scores = {qid: ask_likert(qtext) for qid, qtext in domain_questions[domain].items()}

    general_text = [likert_to_text(personality_questions[k], v) for k, v in general_scores.items()]
    domain_text = [likert_to_text(domain_questions[domain][k], v) for k, v in domain_scores.items()]
    full_text = " ".join(general_text + domain_text)
    return full_text, domain

def predict_personality(text):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = model(encoded["input_ids"], encoded["attention_mask"])
    scores = torch.sigmoid(output).squeeze().tolist()
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    return dict(zip(traits, scores))

def recommend_from_personality(traits, domain):
    top_trait = max(traits, key=traits.get)
    recs = {
        "Books": {
            "Openness": ["The Hobbit", "Dune"],
            "Conscientiousness": ["Atomic Habits"],
            "Extraversion": ["The Alchemist"],
            "Agreeableness": ["Anne of Green Gables"],
            "Neuroticism": ["Quiet", "The Bell Jar"]
        },
        "Movies": {
            "Openness": ["Inception", "The Matrix"],
            "Conscientiousness": ["The Pursuit of Happyness"],
            "Extraversion": ["Mamma Mia!"],
            "Agreeableness": ["Paddington"],
            "Neuroticism": ["Joker", "Inside Out"]
        },
        "Music": {
            "Openness": ["Pink Floyd", "Radiohead"],
            "Conscientiousness": ["Lo-Fi Beats"],
            "Extraversion": ["Pop Party Playlist"],
            "Agreeableness": ["Acoustic Chill"],
            "Neuroticism": ["Sad Songs Playlist"]
        },
        "Games": {
            "Openness": ["The Witcher 3", "Journey"],
            "Conscientiousness": ["Tetris", "Chess"],
            "Extraversion": ["Fortnite", "Overcooked"],
            "Agreeableness": ["Stardew Valley"],
            "Neuroticism": ["Celeste", "Gris"]
        }
    }
    return recs[domain].get(top_trait, ["Explore and discover more!"]), top_trait

def main():
    print("Welcome to the Personality-Based Entertainment Recommender!\n")
    input_text, domain = gather_inputs()
    print("\nAnalyzing your personality...")

    trait_scores = predict_personality(input_text)
    for trait, score in trait_scores.items():
        print(f"{trait}: {score:.2f}")

    recommendations, dominant_trait = recommend_from_personality(trait_scores, domain)

    print(f"\nYour dominant trait is: {dominant_trait}")
    print(f"Recommended {domain} for you:")
    for item in recommendations:
        print(f" - {item}")

if __name__ == "__main__":
    main()
