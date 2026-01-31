import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, log_loss, matthews_corrcoef, cohen_kappa_score,
    hamming_loss, jaccard_score, classification_report, roc_curve, auc
)
import os
import traceback
# Add this line to fix the error:
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

# Path configuration
log_path = r"A:\projects\project3\PythonProject\evaluation_status2.txt"
model_path = r"A:\projects\project3\models\best_bert_lstm2.pth"
plot_dir = r"A:\projects\project3\evaluation\plots2"
results_path = r"A:\projects\project3\evaluation\results2.txt"

# Create directories if they don't exist
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

# Clear previous logs at start
if os.path.exists(log_path):
    os.remove(log_path)


def log_status(message):
    """Log messages to both file and console."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message)


log_status("✅ Evaluation script initialized...")

# === 🔹 CUDA Availability ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_status(
    f"✅ CUDA {'Available! Running on GPU 🚀' if torch.cuda.is_available() else 'Not available. Running on CPU 🖥️'}")
log_status(f"🔹 Evaluation will run on: {device}\n")


# === 1. Define Model Architecture (same as in training) ===
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


# Dataset class (reusing from training)
class PersonalityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]


try:
    # === 2. Load Data ===
    log_status("🔹 Loading dataset...")
    df = pd.read_json(r"A:\projects\project3\data set\processed data\big_five_prompts.jsonl", lines=True)
    df_expanded = df.join(pd.json_normalize(df['response'])).drop(columns=['response']).fillna(0)
    log_status(f"✅ Data loaded! Shape: {df_expanded.shape}")

    # === 3. Get Label Information ===
    personality_traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    log_status(f"✅ Evaluating model for traits: {', '.join(personality_traits)}")

    # We'll need the original labels for classification metrics
    original_labels = {trait: df_expanded[trait].copy() for trait in personality_traits}

    # === 4. Set up tokenizer ===
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # === 5. Load Model ===
    log_status("🔹 Loading best model...")
    model = BERT_LSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    log_status("✅ Model loaded successfully!")

    # === 6. Prepare Test Data ===
    # For simplicity, we'll use the same test split as in the training code
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # Encode labels
    label_encoders = {trait: LabelEncoder() for trait in personality_traits}
    for trait in label_encoders:
        df_expanded[trait] = label_encoders[trait].fit_transform(df_expanded[trait])

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df_expanded['prompt'],
        df_expanded[personality_traits],
        test_size=0.2, random_state=42
    )

    # Create encodings and DataLoader
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128, return_tensors="pt")
    test_loader = DataLoader(
        PersonalityDataset(test_encodings, test_labels),
        batch_size=16,
        shuffle=False
    )
    log_status(f"✅ Test data prepared: {len(test_texts)} samples")

    # === 7. Run Predictions ===
    log_status("🔹 Running predictions...")
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
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

    log_status("✅ Predictions complete!")

    # === 8. Calculate Metrics ===
    log_status("\n🔹 Calculating evaluation metrics...")

    # Initialize results storage
    trait_results = {}
    overall_results = {}

    # Calculate metrics for each personality trait
    for i, trait in enumerate(personality_traits):
        trait_results[trait] = {}

        # Get predictions and true values for this trait
        trait_pred = y_pred[:, i]
        trait_true = y_true[:, i]

        # Convert to class labels for classification metrics
        trait_pred_class = np.round(trait_pred).astype(int)
        trait_true_class = trait_true.astype(int)

        # Number of classes for this trait
        num_classes = len(np.unique(trait_true_class))

        # Calculate metrics
        trait_results[trait]['accuracy'] = accuracy_score(trait_true_class, trait_pred_class)

        precision, recall, f1, _ = precision_recall_fscore_support(
            trait_true_class, trait_pred_class, average='weighted'
        )
        trait_results[trait]['precision'] = precision
        trait_results[trait]['recall'] = recall
        trait_results[trait]['f1'] = f1

        # Calculate MCC
        trait_results[trait]['mcc'] = matthews_corrcoef(trait_true_class, trait_pred_class)

        # Calculate Kappa
        trait_results[trait]['kappa'] = cohen_kappa_score(trait_true_class, trait_pred_class)

        # Calculate Hamming Loss
        trait_results[trait]['hamming_loss'] = hamming_loss(trait_true_class, trait_pred_class)

        # Calculate Jaccard Score
        trait_results[trait]['jaccard'] = jaccard_score(
            trait_true_class, trait_pred_class, average='weighted', zero_division=0
        )

        # ROC AUC (one-vs-rest for multi-class)
        if num_classes > 2:
            try:
                trait_results[trait]['roc_auc'] = roc_auc_score(
                    np.eye(num_classes)[trait_true_class],
                    np.eye(num_classes)[trait_pred_class],
                    multi_class='ovr',
                    average='weighted'
                )
            except ValueError:
                trait_results[trait]['roc_auc'] = "N/A (Need probability estimates)"
        else:
            try:
                trait_results[trait]['roc_auc'] = roc_auc_score(trait_true_class, trait_pred_class)
            except ValueError:
                trait_results[trait]['roc_auc'] = "N/A"

        # Log Loss
        try:
            # Convert to one-hot for multi-class log_loss
            trait_results[trait]['log_loss'] = log_loss(
                np.eye(num_classes)[trait_true_class],
                np.eye(num_classes)[trait_pred_class]
            )
        except ValueError:
            trait_results[trait]['log_loss'] = "N/A (Need probability estimates)"

        # Calculate confusion matrix
        trait_results[trait]['confusion_matrix'] = confusion_matrix(trait_true_class, trait_pred_class)

    # Calculate overall metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average='micro'
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average='weighted'
    )

    overall_results = {
        'accuracy': accuracy,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'mcc': matthews_corrcoef(y_true_classes, y_pred_classes),
        'kappa': cohen_kappa_score(y_true_classes, y_pred_classes),
        'hamming_loss': hamming_loss(y_true_classes, y_pred_classes),
    }

    log_status("✅ Metrics calculation complete!")

    # === 9. Output Results ===
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== PERSONALITY MODEL EVALUATION RESULTS ===\n\n")

        # Overall results
        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        for metric, value in overall_results.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")

        # Per trait results
        f.write("PER-TRAIT METRICS\n")
        f.write("-" * 40 + "\n")
        for trait, metrics in trait_results.items():
            f.write(f"\n{trait.upper()}\n")
            f.write("-" * 20 + "\n")
            for metric, value in metrics.items():
                if metric != 'confusion_matrix':
                    if isinstance(value, str):
                        f.write(f"{metric}: {value}\n")
                    else:
                        f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")

    log_status(f"✅ Results saved to {results_path}")

    # === 10. Generate Visualizations ===
    log_status("\n🔹 Generating visualizations...")

    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Confusion Matrix for each trait
    for trait, metrics in trait_results.items():
        plt.figure(figsize=(8, 6))
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(len(cm)),
                    yticklabels=range(len(cm)))
        plt.title(f'Confusion Matrix - {trait}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{trait}.png"))
        plt.close()

    # 2. Per-trait performance comparison
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    plt.figure(figsize=(12, 8))

    # Create data for plotting
    plot_data = []
    for trait, metrics in trait_results.items():
        for metric in metrics_to_plot:
            if metric in metrics and not isinstance(metrics[metric], str):
                plot_data.append({
                    'Trait': trait,
                    'Metric': metric,
                    'Value': metrics[metric]
                })

    plot_df = pd.DataFrame(plot_data)

    # Plot
    sns.barplot(x='Trait', y='Value', hue='Metric', data=plot_df)
    plt.title('Performance Metrics by Personality Trait')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "trait_performance_comparison.png"))
    plt.close()

    # 3. ROC Curves (if applicable)
    # This would typically require probability outputs, but we'll set up the framework
    for i, trait in enumerate(personality_traits):
        try:
            # This is a simplified version - in a real scenario, you'd need proper probability outputs
            trait_pred = y_pred[:, i]
            trait_true = y_true[:, i].astype(int)

            # Number of unique classes
            classes = np.unique(trait_true)
            n_classes = len(classes)

            if n_classes > 2:  # Multi-class
                # One-vs-Rest approach
                plt.figure(figsize=(10, 8))

                for cls in classes:
                    binary_true = (trait_true == cls).astype(int)
                    binary_pred = (trait_pred == cls).astype(float)

                    fpr, tpr, _ = roc_curve(binary_true, binary_pred)
                    roc_auc = auc(fpr, tpr)

                    plt.plot(fpr, tpr, lw=2,
                             label=f'Class {cls} (AUC = {roc_auc:.2f})')

                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve (One-vs-Rest) - {trait}')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(plot_dir, f"roc_curve_{trait}.png"))
                plt.close()

            elif n_classes == 2:  # Binary
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(trait_true, trait_pred)
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {trait}')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(plot_dir, f"roc_curve_{trait}.png"))
                plt.close()

        except Exception as e:
            log_status(f"⚠️ Could not generate ROC curve for {trait}: {str(e)}")

    # 4. Overall metrics comparison
    plt.figure(figsize=(14, 6))
    metrics_to_compare = [
        ('precision', ['precision_micro', 'precision_macro', 'precision_weighted']),
        ('recall', ['recall_micro', 'recall_macro', 'recall_weighted']),
        ('f1', ['f1_micro', 'f1_macro', 'f1_weighted'])
    ]

    bar_width = 0.25
    index = np.arange(3)  # precision, recall, f1

    for i, (metric_type, variants) in enumerate(metrics_to_compare):
        for j, (variant_suffix, variant_name) in enumerate([
            ('_micro', 'Micro'), ('_macro', 'Macro'), ('_weighted', 'Weighted')
        ]):
            plt.bar(
                index[j] + i * bar_width,
                overall_results[f"{metric_type}{variant_suffix}"],
                bar_width,
                label=f"{variant_name}"
            )

    plt.xlabel('Metric Type')
    plt.ylabel('Score')
    plt.title('Comparison of Micro, Macro and Weighted Metrics')
    plt.xticks(index + bar_width, ['Precision', 'Recall', 'F1'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "overall_metrics_comparison.png"))
    plt.close()

    log_status(f"✅ Visualizations saved to {plot_dir}")
    log_status("\n✅ Evaluation completed successfully! 🎉")

except Exception as e:
    log_status(f"❌ Error during evaluation: {str(e)}\n{traceback.format_exc()}")