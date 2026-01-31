import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r'A:\projects\project3\PythonProject\processed_bigfive.csv')

# Drop completely missing columns
df.drop(columns=['dateload', 'country'], inplace=True)
print(df.shape)  # Outputs (rows, columns)

# Detecting and handling outliers using the IQR method
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    return df
print(df.shape)  # Outputs (rows, columns)

# Identify personality trait columns
trait_columns = df.columns[df.columns.str.startswith(('EXT', 'EST', 'AGR', 'CSN', 'OPN'))]
df = remove_outliers_iqr(df, trait_columns)
print(df.shape)  # Outputs (rows, columns)

# Handling missing values using KNN imputation
imputer = KNNImputer(n_neighbors=5)
df[trait_columns] = imputer.fit_transform(df[trait_columns])
print(df.shape)  # Outputs (rows, columns)

# Standardizing the data
scaler = StandardScaler()
df[trait_columns] = scaler.fit_transform(df[trait_columns])
print(df.shape)  # Outputs (rows, columns)

# Save the cleaned dataset
df.to_csv(r'A:\projects\project3\data set\big five\cleaned_data.csv', index=False)


print("Preprocessing completed! Cleaned dataset saved.")
print(df.shape)  # Outputs (rows, columns)
