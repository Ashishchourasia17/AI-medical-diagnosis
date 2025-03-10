import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df.fillna(df.mean(), inplace=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(df.iloc[:, :-1])  # Scale all features except the target column
    y = df.iloc[:, -1]  # Target column (Outcome)
    return X, y
