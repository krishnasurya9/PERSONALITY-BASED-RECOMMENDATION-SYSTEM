import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, StratifiedKFold
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import os
import traceback
import time

# Path configuration
log_path = r"A:\projects\project3\PythonProject\evaluation_status.txt"
model_path = r"A:\projects\project3\models\best_bert_lstm.pth"
plot_dir = r"A:\projects\project3\evaluation\plots"
results_path = r"A:\projects\project3\evaluation\results.txt"

# Create directories if they don't exist
os.makedirs(plot_dir, exist_ok=True)


# Logging setup
def log_status(message):
    """Log messages to both file and console."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message)


if os.path.exists(log_path):
    os.remove(log_path)

log_status("✅ Validation analysis initialized...")

# CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_status(f"✅ Using device: {device}")


# Dataset and model definitions
class PersonalityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings

        # Convert categorical labels to numeric values first
        # Create a mapping for each trait
        label_maps = {}
        numeric_labels = labels.copy()

        for trait in labels.columns:
            unique_values = labels[trait].unique()
            label_map = {val: idx for idx, val in enumerate(unique_values)}
            label_maps[trait] = label_map

            # Convert to numeric
            numeric_labels[trait] = labels[trait].map(label_map)

        # Now convert to tensor
        self.labels = torch.tensor(numeric_labels.values, dtype=torch.float32)
        self.label_maps = label_maps  # Store for reference

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]


class BERT_LSTM(torch.nn.Module):
    def __init__(self):
        super(BERT_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = torch.nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(256 * 2, 5)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        lstm_out, _ = self.lstm(bert_outputs.last_hidden_state)
        return self.fc(lstm_out[:, -1, :])


# Simpler model for comparison
class SimpleBERTClassifier(torch.nn.Module):
    def __init__(self):
        super(SimpleBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.fc(x)


try:
    # Load data
    log_status("🔹 Loading dataset...")
    df = pd.read_json(r"A:\projects\project3\data set\processed data\big_five_prompts.jsonl", lines=True)
    df_expanded = df.join(pd.json_normalize(df['response'])).drop(columns=['response']).fillna(0)
    log_status(f"✅ Data loaded! Shape: {df_expanded.shape}")

    # Define personality traits
    personality_traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

    # Setup tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 1. DATA LEAKAGE ANALYSIS
    log_status("\n=== 1. DATA LEAKAGE ANALYSIS ===")

    # Check for duplicates in the dataset
    duplicate_count = df_expanded.duplicated(subset=['prompt']).sum()
    log_status(f"Duplicate prompts in dataset: {duplicate_count} ({duplicate_count / len(df_expanded) * 100:.2f}%)")

    # Check distribution of labels
    log_status("\nLabel distribution:")
    for trait in personality_traits:
        value_counts = df_expanded[trait].value_counts()
        log_status(f"{trait}: {value_counts.to_dict()}")

    # 2. CROSS-VALIDATION
    log_status("\n=== 2. CROSS-VALIDATION ===")


    # Function to evaluate on a batch
    def evaluate_model(model, data_loader, device):
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = {k: v.to(device) for k, v in batch[0].items()}, batch[1].to(device)
                outputs = model(**inputs)

                # Convert to numpy for sklearn metrics
                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Combine batches
        y_pred = np.vstack(all_predictions)
        y_true = np.vstack(all_labels)

        # Convert predictions to class labels
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_classes, y_pred_classes, average='weighted', zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


    # Perform k-fold validation
    def perform_kfold(df, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []

        # Get all texts and labels
        texts = df['prompt']
        labels = df[personality_traits]

        for fold, (train_idx, test_idx) in enumerate(kf.split(texts)):
            log_status(f"\nFold {fold + 1}/{n_splits}")

            # Split data
            train_texts, test_texts = texts.iloc[train_idx], texts.iloc[test_idx]
            train_labels, test_labels = labels.iloc[train_idx], labels.iloc[test_idx]

            # Create encodings
            test_encodings = tokenizer(
                list(test_texts),
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )

            # Create DataLoader
            test_loader = DataLoader(
                PersonalityDataset(test_encodings, test_labels),
                batch_size=16,
                shuffle=False
            )

            # Load model with fresh weights for each fold
            model = BERT_LSTM().to(device)

            # Initialize with random weights for this test
            # We're testing data leakage by using an untrained model
            # If performance is still high with random weights, we have a problem

            # Evaluate model
            log_status("Evaluating untrained model (random weights)...")
            metrics = evaluate_model(model, test_loader, device)
            log_status(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

            fold_results.append(metrics)

        # Calculate average metrics
        avg_metrics = {
            metric: np.mean([fold[metric] for fold in fold_results])
            for metric in ['accuracy', 'precision', 'recall', 'f1']
        }

        return avg_metrics, fold_results


    # Run k-fold validation with untrained model
    log_status("\nRunning 5-fold cross-validation with UNTRAINED model (random weights)")
    log_status("This tests for data leakage - if accuracy is high with random weights, there's a problem")

    # Use a small subset for quick validation
    sample_size = min(10000, len(df_expanded))
    df_sample = df_expanded.sample(sample_size, random_state=42)

    untrained_avg_metrics, untrained_fold_results = perform_kfold(df_sample)

    log_status("\nUntrained model average metrics across 5 folds:")
    for metric, value in untrained_avg_metrics.items():
        log_status(f"{metric}: {value:.4f}")

    # 3. SIMPLER MODEL COMPARISON
    log_status("\n=== 3. SIMPLER MODEL COMPARISON ===")


    # Function to train a simple model on a fold
    def train_and_evaluate_simple_model(train_texts, train_labels, test_texts, test_labels):
        # Create encodings
        train_encodings = tokenizer(
            list(train_texts),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        test_encodings = tokenizer(
            list(test_texts),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        # Create DataLoaders
        train_loader = DataLoader(
            PersonalityDataset(train_encodings, train_labels),
            batch_size=16,
            shuffle=True
        )
        test_loader = DataLoader(
            PersonalityDataset(test_encodings, test_labels),
            batch_size=16,
            shuffle=False
        )

        # Initialize simple model
        model = SimpleBERTClassifier().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = torch.nn.MSELoss()

        # Train for just a few steps to see if patterns are easily learnable
        log_status("Training simple model for 5 steps...")
        model.train()
        for i, batch in enumerate(train_loader):
            if i >= 5:  # Just 5 training steps
                break

            inputs, labels = {k: v.to(device) for k, v in batch[0].items()}, batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            log_status(f"Step {i + 1}, Loss: {loss.item():.4f}")

        # Evaluate
        model.eval()
        metrics = evaluate_model(model, test_loader, device)
        return metrics


    # Perform simple model training and evaluation
    log_status("\nTraining and evaluating a simpler BERT model")

    # Split data
    from sklearn.model_selection import train_test_split

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df_sample['prompt'],
        df_sample[personality_traits],
        test_size=0.2,
        random_state=42
    )

    simple_metrics = train_and_evaluate_simple_model(
        train_texts, train_labels, test_texts, test_labels
    )

    log_status("\nSimple model metrics after minimal training:")
    for metric, value in simple_metrics.items():
        log_status(f"{metric}: {value:.4f}")

    # 4. DATA INTEGRITY CHECK
    log_status("\n=== 4. DATA INTEGRITY CHECK ===")

    # Check if the labels are too perfectly aligned with inputs
    # For example, check if the same prompt always has the same labels
    prompt_groups = df_expanded.groupby('prompt')[personality_traits].nunique()
    consistent_prompts = (prompt_groups == 1).all(axis=1).sum()

    log_status(
        f"Prompts with 100% consistent labels: {consistent_prompts}/{len(prompt_groups)} ({consistent_prompts / len(prompt_groups) * 100:.2f}%)")

    # Check if there's pattern in the prompts that perfectly predicts the labels
    # Look for most common words in prompts for each trait level
    from collections import Counter
    import re


    def analyze_word_distribution(trait):
        word_counters = {}
        for value in df_expanded[trait].unique():
            # Get prompts for this trait value
            prompts = df_expanded[df_expanded[trait] == value]['prompt']

            # Tokenize and count words
            all_words = []
            for prompt in prompts:
                # Simple tokenization - split by non-alphanumeric chars
                words = re.findall(r'\b\w+\b', prompt.lower())
                all_words.extend(words)

            word_counters[value] = Counter(all_words).most_common(10)

        return word_counters


    log_status("\nAnalyzing common words by trait value (top 10):")
    for trait in personality_traits[:1]:  # Just analyze one trait for brevity
        log_status(f"\n{trait}:")
        word_dist = analyze_word_distribution(trait)
        for value, top_words in word_dist.items():
            log_status(f"  Value {value}: {top_words}")

    # 5. TEST SET VS MODEL PREDICTION ANALYSIS
    log_status("\n=== 5. TEST SET VS MODEL PREDICTION ANALYSIS ===")

    # Create a small test set
    test_size = min(100, len(df_expanded))
    test_df = df_expanded.sample(test_size, random_state=42)
    test_texts = test_df['prompt']
    test_labels = test_df[personality_traits]

    # Tokenize
    test_encodings = tokenizer(
        list(test_texts),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    # Create DataLoader
    test_loader = DataLoader(
        PersonalityDataset(test_encodings, test_labels),
        batch_size=16,
        shuffle=False
    )

    # Load the original trained model
    try:
        log_status("\nLoading trained model...")
        trained_model = BERT_LSTM().to(device)
        trained_model.load_state_dict(torch.load(model_path, map_location=device))
        log_status("Trained model loaded successfully!")

        # Evaluate trained model
        trained_metrics = evaluate_model(trained_model, test_loader, device)
        log_status("\nTrained model metrics on test set:")
        for metric, value in trained_metrics.items():
            log_status(f"{metric}: {value:.4f}")

        # Compare with random model
        random_model = BERT_LSTM().to(device)  # Fresh model with random weights
        random_metrics = evaluate_model(random_model, test_loader, device)
        log_status("\nRandom model metrics on same test set:")
        for metric, value in random_metrics.items():
            log_status(f"{metric}: {value:.4f}")

        # Show the difference
        log_status("\nPerformance difference (trained - random):")
        for metric in trained_metrics:
            diff = trained_metrics[metric] - random_metrics[metric]
            log_status(f"{metric} difference: {diff:.4f}")

    except Exception as e:
        log_status(f"Error loading trained model: {str(e)}")

    # 6. FEATURE IMPORTANCE ANALYSIS
    log_status("\n=== 6. FEATURE IMPORTANCE ANALYSIS ===")

    # Select a few examples to analyze
    sample_texts = test_texts.iloc[:5].tolist()
    sample_labels = test_labels.iloc[:5].values


    # Function to analyze token importance via occlusion
    def analyze_token_importance(model, text, label):
        # Tokenize text
        tokens = tokenizer.tokenize(text)
        token_importance = []

        # Original prediction
        encoding = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            original_output = model(**inputs).cpu().numpy()[0]
            original_pred = np.argmax(original_output)

        # Mask each token and see impact
        for i in range(len(tokens)):
            # Skip special tokens
            if tokens[i] in ['[CLS]', '[SEP]', '[PAD]']:
                token_importance.append(0)
                continue

            # Create masked text
            masked_tokens = tokens.copy()
            masked_tokens[i] = '[MASK]'
            masked_text = tokenizer.convert_tokens_to_string(masked_tokens)

            # Get prediction on masked text
            encoding = tokenizer(
                masked_text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in encoding.items()}

            with torch.no_grad():
                masked_output = model(**inputs).cpu().numpy()[0]
                masked_pred = np.argmax(masked_output)

            # Calculate importance as change in prediction
            importance = np.abs(original_output - masked_output).sum()
            token_importance.append(importance)

        # Normalize importance
        if max(token_importance) > 0:
            token_importance = [i / max(token_importance) for i in token_importance]

        return list(zip(tokens, token_importance))


    # Try to analyze at least one example
    try:
        if len(sample_texts) > 0:
            log_status("\nAnalyzing token importance for a sample text:")
            importance = analyze_token_importance(trained_model, sample_texts[0], sample_labels[0])

            # Display top important tokens
            sorted_importance = sorted(importance, key=lambda x: x[1], reverse=True)
            log_status(f"Text: {sample_texts[0]}")
            log_status("Top 10 important tokens:")
            for token, imp in sorted_importance[:10]:
                log_status(f"  {token}: {imp:.4f}")
    except Exception as e:
        log_status(f"Error in token importance analysis: {str(e)}")

    # 7. SUMMARY AND RECOMMENDATIONS
    log_status("\n=== 7. SUMMARY AND RECOMMENDATIONS ===")
    log_status("""
Based on the validation analysis, here are possible explanations for the high performance:

1. Data characteristics:
   - Is there a strong correlation between specific words/patterns and personality traits?
   - Are the labels too consistent for identical prompts?

2. Model behavior:
   - How does the trained model compare to random initialization?
   - Does a simpler model achieve similar results with minimal training?

3. Cross-validation reliability:
   - Are results consistent across different data splits?
   - Does random weight initialization still yield high accuracy?

Recommended next steps:
1. If trained model significantly outperforms random, the model is learning real patterns
2. If the simpler model achieves similar results, consider using it for efficiency
3. If duplicate/correlated data is found, clean the dataset and retest
4. Consider external validation with a completely different dataset
5. Implement additional regularization techniques if overfitting is suspected
""")

    log_status("\n✅ Validation analysis completed successfully!")

except Exception as e:
    log_status(f"❌ Error during validation analysis: {str(e)}\n{traceback.format_exc()}")