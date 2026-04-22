# Import necessary Libraries
import pandas as pd
import os

# Import the dataset
# Construct the path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'employee_attrition.csv')
df= pd.read_csv(data_path)
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())
# Get the shape of the dataset
print("\nShape of the dataset:")
print(df.shape)

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())
print("\nNOTE: No missing values found in the dataset.")

# Project requires to have some missing values to demonstrate handling them, so we will artificially introduce some missing values in the 'Age' column for demonstration purposes.
import numpy as np
# Randomly select 10% of the rows to have missing values in the 'Age' column
num_missing = int(0.1 * len(df))
missing_indices = np.random.choice(df.index, num_missing, replace=False)
df.loc[missing_indices, 'Age'] = np.nan
print("\nIntroduced missing values in the 'Age' column for demonstration purposes.")
print("\nMissing values in each column after introducing missing values:")
print(df.isnull().sum())

# Get the data types of each column
print("\nData types of each column:")
print(df.dtypes)

# Get summary statistics of the dataset
print("\nSummary statistics of the dataset:")
print(df.describe())

# Check the distribution of the target variable 'Attrition'
print("\nDistribution of the target variable 'Attrition':")
print(df['Attrition'].value_counts())

# Encode the target variable 'Attrition' (Yes=1, No=0)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
# Change the data type of 'Attrition' to integer
df['Attrition'] = df['Attrition'].astype(int)

# Display the first few rows after encoding
print("\nFirst few rows after encoding Attrition column:")
print(df.head())

# Select features and target variable
features = df.drop('Attrition', axis=1)
target = df['Attrition']
# Display the features and target variable
print("\nSelected features and target variable:")
print("\nFeatures:")
print(features.head())
print("\nTarget variable:")
print(target.head())

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print("\nData split into training and testing sets.")
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

# Handle missing values in the 'Age' column by filling them with the mean age from training data
mean_age = X_train['Age'].mean()
X_train['Age'] = X_train['Age'].fillna(mean_age)
X_test['Age'] = X_test['Age'].fillna(mean_age)
print("\nHandled missing values in the 'Age' column by filling them with the mean age from training data.")
print("\nMissing values in training set after handling:")
print(X_train.isnull().sum())
print("\nMissing values in testing set after handling:")
print(X_test.isnull().sum())

# Encode categorical variables using one-hot encoding
categorical_cols = X_train.select_dtypes(include=['object']).columns
print(f"\nCategorical columns to encode: {list(categorical_cols)}")

X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
print("\nApplied one-hot encoding to categorical variables.")
print(f"Training set shape after encoding: {X_train.shape}")
print(f"Testing set shape after encoding: {X_test.shape}")

# Ensure both sets have the same columns (in case of missing categories in test)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
print("\nAligned training and testing sets to have the same columns.")

# Display the first few rows of the preprocessed training set
print("\nFirst few rows of preprocessed training features:")
print(X_train.head())  

