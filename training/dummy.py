import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# === Questions ===
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
    }
}

# === Dummy answers ===
general_scores = {
    "Openness": 5,
    "Conscientiousness": 4,
    "Extraversion": 3,
    "Agreeableness": 4,
    "Neuroticism": 2
}

domain_responses = {
    "Q1": 5,
    "Q2": 4,
    "Q3": 3
}

# === Combine domain responses into text format ===
domain_input_text = " ".join([f"{k}:{v}" for k, v in domain_responses.items()])

# === Model class ===
class BERTLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=3):
        super(BERTLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        model = BERTLSTMClassifier(num_classes=5)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)
        last_hidden_state = lstm_output[:, -1, :]
        logits = self.fc(last_hidden_state)
        return logits

# === Load model and tokenizer ===
device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BERTLSTMClassifier()
model.load_state_dict(torch.load(r"A:\projects\project3\models\best_bert_lstm.pth", map_location=device))
model.to(device)
model.eval()

# === Tokenize and predict ===
inputs = tokenizer(domain_input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    logits = model(**inputs)
    prediction = torch.argmax(logits, dim=1).item()

print("=== Personality Summary ===")
for trait, score in general_scores.items():
    print(f"{trait}: {score}")

print("\n=== Domain Responses (Books) ===")
for qid, ans in domain_responses.items():
    print(f"{domain_specific_questions['Books'][qid]} -> {ans}")

print("\n=== Model Output ===")
print("Logits:", logits)
print("Predicted class:", prediction)
