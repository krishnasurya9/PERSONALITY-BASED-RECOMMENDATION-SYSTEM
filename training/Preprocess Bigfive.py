import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = r"A:\projects\project3\data set\big five\data.csv"  # Update if needed
data = pd.read_csv(file_path, delimiter="\t")  # Use '\t' if data is tab-separated

# Check column names
print("Columns in dataset:", data.columns)

# If the columns are in a single string (due to incorrect delimiter), split them
if len(data.columns) == 1:
    data = data.iloc[:, 0].str.split("\t", expand=True)
    print("Columns have been split correctly.")

# Renaming columns (if needed)
expected_columns = ["EXT1", "EXT2", "EXT3", "EXT4", "EXT5", "EXT6", "EXT7", "EXT8", "EXT9", "EXT10",
                    "EST1", "EST2", "EST3", "EST4", "EST5", "EST6", "EST7", "EST8", "EST9", "EST10",
                    "AGR1", "AGR2", "AGR3", "AGR4", "AGR5", "AGR6", "AGR7", "AGR8", "AGR9", "AGR10",
                    "CSN1", "CSN2", "CSN3", "CSN4", "CSN5", "CSN6", "CSN7", "CSN8", "CSN9", "CSN10",
                    "OPN1", "OPN2", "OPN3", "OPN4", "OPN5", "OPN6", "OPN7", "OPN8", "OPN9", "OPN10"]
data.columns = expected_columns + list(data.columns[len(expected_columns):])  # Preserve extra columns

# Convert all values to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Compute Big Five trait scores (averaging the relevant columns)
data["Extraversion"] = data[[f"EXT{i}" for i in range(1, 11)]].mean(axis=1)
data["Neuroticism"] = data[[f"EST{i}" for i in range(1, 11)]].mean(axis=1)
data["Agreeableness"] = data[[f"AGR{i}" for i in range(1, 11)]].mean(axis=1)
data["Conscientiousness"] = data[[f"CSN{i}" for i in range(1, 11)]].mean(axis=1)
data["Openness"] = data[[f"OPN{i}" for i in range(1, 11)]].mean(axis=1)

# Drop raw question columns (optional, keep only the final traits)
drop_columns = expected_columns  # Remove individual question columns
data = data.drop(columns=drop_columns, errors='ignore')

# Normalize the computed traits using StandardScaler
scaler = StandardScaler()
trait_columns = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
data[trait_columns] = scaler.fit_transform(data[trait_columns])

# Save the preprocessed dataset
processed_file_path = r"E:\projects\project3\PythonProject\processed_bigfive.csv"
data.to_csv(processed_file_path, index=False)

print("Preprocessing completed successfully! Processed data saved at:", processed_file_path)
