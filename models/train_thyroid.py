import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/thyroid.csv")
MODEL_PATH = os.path.join(BASE_DIR, "thyroid_model.sav")

# Check if dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please add 'thyroid.csv' in the 'data' folder.")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Print dataset columns for debugging
print("Dataset columns:", df.columns)

# Ensure the dataset has the correct columns based on available data
required_columns = ["age", "sex", "on_thyroxine", "TSH", "T3_measured", "T3", "TT4", "binaryClass"]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing expected columns: {missing_columns}")

# Preprocess data
X = df.drop(columns=["binaryClass"])  # Features
y = df["binaryClass"]  # Target variable

# Encode categorical features if any
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':  # Check for non-numeric columns
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])

# Print feature count to debug mismatches
print("Number of features used in training:", X.shape[1])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model, model_file)

print(f"Thyroid model saved successfully at: {MODEL_PATH}")
