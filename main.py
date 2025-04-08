import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# LSTM Model
class XSSDetectorLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim=50, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x.unsqueeze(1))
        x = self.fc(x[:, -1, :])
        return x

# MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)

# Model selector
def load_model(model_name, input_dim):
    if model_name == "CNN":
        return CNNModel(input_dim)
    elif model_name == "LSTM":
        return XSSDetectorLSTM(input_dim)
    elif model_name == "MLP":
        return MLPModel(input_dim)
    else:
        raise ValueError("Unknown model name")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XSS Detection Model")
    parser.add_argument("-i", "--input", required=True, help="Input file containing the text to be analyzed")
    parser.add_argument("-m", "--model", required=True, choices=["CNN", "LSTM", "MLP"], help="Model type")
    parser.add_argument("-w", "--weight", required=True, help="Model weight file (.pth)")
    parser.add_argument("-o", "--output", required=True, help="Output file to save the result")
    args = parser.parse_args()

    # Prepare vectorizer
    VECTORIZER_PATH = "vectorizer.pkl"
    DATASET_PATH = "Training Dataset/final_dataset.csv"

    if os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        df = pd.read_csv(DATASET_PATH)
        df = df.dropna(subset=["Sentence"])
        df = df[df["Sentence"].apply(lambda x: isinstance(x, str))]
        X_train = df["Sentence"]
        vectorizer = TfidfVectorizer()
        vectorizer.fit(X_train)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)

    # Read input
    with open(args.input, "r") as f:
        input_lines = [line.strip() for line in f.readlines() if line.strip()]

    input_dim = len(vectorizer.transform([input_lines[0]]).toarray()[0])
    model = load_model(args.model, input_dim)
    model.load_state_dict(torch.load(args.weight, map_location=torch.device("cpu")), strict=False)
    model.eval()

    results = []
    for line in input_lines:
        input_data = vectorizer.transform([line]).toarray()
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            confidence = prob[0][pred].item()
            label = "Malicious" if pred == 1 else "Normal"
            results.append(f"{label} ({confidence:.4f})")

    with open(args.output, "w") as f:
        for r in results:
            f.write(r + "\n")

    print(f"Prediction results saved to {args.output}")
