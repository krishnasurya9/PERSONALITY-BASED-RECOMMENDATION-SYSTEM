import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import traceback

log_path = r"A:\projects\project3\PythonProject\status.txt"
def log_status(message):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message)

# Clear previous logs at start
if os.path.exists(log_path):
    os.remove(log_path)

log_status("✅ Logging initialized. Training started...\n")

# === 🔹 CUDA Availability ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_status(f"✅ CUDA {'Available! Running on GPU 🚀' if torch.cuda.is_available() else 'Not available. Running on CPU 🖥️'}")
log_status(f"🔹 Model will run on: {device}\n")

# === 1. Load Data ===
log_status("🔹 Loading dataset...")
df = pd.read_json(r"A:\projects\project3\data set\processed data\big_five_prompts.jsonl", lines=True)
df_expanded = df.join(pd.json_normalize(df['response'])).drop(columns=['response']).fillna(0)
log_status(f"✅ Data loaded! Shape: {df_expanded.shape}")

# === 2. Encode Labels ===
log_status("🔹 Encoding labels...")
label_encoders = {trait: LabelEncoder() for trait in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]}
for trait in label_encoders:
    df_expanded[trait] = label_encoders[trait].fit_transform(df_expanded[trait])
log_status("✅ Label encoding complete!")

# === 3. Split Data ===
log_status("🔹 Splitting dataset...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_expanded['prompt'],
    df_expanded[["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]],
    test_size=0.2, random_state=42)
log_status(f"✅ Train set: {len(train_texts)}, Test set: {len(test_texts)}")

# === 4. Tokenization ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_batch(texts):
    return tokenizer(list(texts), truncation=True, padding=True, max_length=128, return_tensors="pt")

train_encodings, test_encodings = tokenize_batch(train_texts), tokenize_batch(test_texts)
log_status("✅ Tokenization complete!")

# === 5. Dataset & DataLoader ===
class PersonalityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

batch_size = 16
train_loader = DataLoader(PersonalityDataset(train_encodings, train_labels), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(PersonalityDataset(test_encodings, test_labels), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
log_status("✅ DataLoader ready!")

# === 6. Define Optimized BERT-LSTM Model ===
class BERT_LSTM(nn.Module):
    def __init__(self, freeze_bert=True):
        super(BERT_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256 * 2, 5)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        lstm_out, _ = self.lstm(bert_outputs.last_hidden_state)
        return self.fc(lstm_out[:, -1, :])

# === 7. Initialize Model, Optimizer & Loss ===
model = BERT_LSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
scaler = torch.cuda.amp.GradScaler()
log_status(f"✅ Model successfully moved to: {next(model.parameters()).device}")

# === 8. Training & Validation Loop ===
num_epochs = 5
best_val_loss = float('inf')
best_model_path = r"A:\projects\project3\models\best_bert_lstm.pth"

log_status("\n🟡 Starting training...\n")

try:
    for epoch in range(num_epochs):
        log_status(f"🟡 Epoch {epoch + 1} starting...")
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            inputs, labels = {k: v.to(device) for k, v in batch[0].items()}, batch[1].to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        log_status(f"✅ Epoch {epoch + 1} Training Loss: {train_loss:.4f}")

        # === Validation Phase ===
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = {k: v.to(device) for k, v in batch[0].items()}, batch[1].to(device)
                outputs = model(**inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(test_loader)
        log_status(f"✅ Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            log_status(f"🌟 Best model saved with Validation Loss: {val_loss:.4f}")

    log_status(f"\n✅ Training completed! Best model saved at {best_model_path} with Validation Loss: {best_val_loss:.4f}")

except Exception as e:
    log_status(f"❌ Error occurred: {str(e)}\n{traceback.format_exc()}")
