import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from .config import MODEL_PATH

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

# Global instances
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None
model = None

def load_ai_model():
    """Loads the model and tokenizer into global variables."""
    global model, tokenizer
    try:
        print("... Loading tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        print("... Loading model...")
        model = BERT_LSTM()
        # Load state dict with map_location to handle CPU/GPU differences
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("[OK] Model and tokenizer loaded successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return False

def get_model():
    return model, tokenizer, device
