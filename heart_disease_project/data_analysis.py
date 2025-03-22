
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Load Dataset
def load_dataset(filename):
    """Loads the dataset if it exists."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")
    return pd.read_csv(filename)

# Handle Missing Values
def handle_missing_values(df):
    """Fills missing values with median for numerical columns and mode for categorical columns."""
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# Normalize Data
def normalize_data(df, numerical_features):
    """Scales numerical features using MinMaxScaler."""
    existing_features = [col for col in numerical_features if col in df.columns]
    if not existing_features:
        raise ValueError("No valid numerical features found!")
    
    scaler = MinMaxScaler()
    df[existing_features] = scaler.fit_transform(df[existing_features])
    return df

# Encode Categorical Variables
def encode_categorical(df, categorical_features):
    """Encodes categorical features using One-Hot Encoding."""
    df = pd.get_dummies(df, columns=[col for col in categorical_features if col in df.columns], drop_first=True)
    return df

# Feature Selection
def select_top_features(df, target_column="target", top_n=10):
    """Selects top features based on correlation with the target column."""
    corr = df.corr()[target_column].abs().sort_values(ascending=False)
    return corr.index[1:top_n + 1].tolist()

# Data Visualization
def visualize_data(df):
    """Generates visualizations for the dataset."""
    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Histograms
    df.hist(figsize=(12, 8), bins=20, edgecolor='black')
    plt.suptitle("Feature Distributions")
    plt.show()

    # Boxplots
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=45)
    plt.title("Boxplot of Features")
    plt.show()

# Feature Importance Analysis
def feature_importance_analysis(df, target_column="target"):
    """Analyzes and visualizes feature importance using RandomForestClassifier."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    feature_importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 5))
    plt.title("Feature Importance for Heart Disease Prediction")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.show()

# Save Cleaned Data
def save_cleaned_data(df, filename="cleaned_data.csv"):
    """Saves the processed dataset to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Cleaned data saved as {filename}")

# Main Function
def main():
    # Step 1: Dataset Processing
    filename = "heart.csv"
    df = load_dataset(filename)
    
    numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    
    df = handle_missing_values(df)
    df = normalize_data(df, numerical_features)
    df = encode_categorical(df, categorical_features)

    # Feature Selection
    top_features = select_top_features(df, target_column="target", top_n=10)
    print("Top Features:", top_features)
    df = df[top_features + ["target"]]

    # Save Cleaned Data
    save_cleaned_data(df)

    # Step 2: Data Visualization
    cleaned_data_path = "cleaned_data.csv"
    df_cleaned = pd.read_csv(cleaned_data_path)

    print("Statistical Summary:")
    print(df_cleaned.describe())

    visualize_data(df_cleaned)
    feature_importance_analysis(df_cleaned)

if __name__ == "__main__":
    main()