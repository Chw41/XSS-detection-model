import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# CNN
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# LSTM
class XSSDetectorLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        x = self.fc(hn[-1])
        return x

# MLP
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

# 模型選擇
def load_model(model_name, input_dim=None, vocab_size=None):
    if model_name == "CNN":
        return CNNModel()
    elif model_name == "LSTM":
        if vocab_size is None:
            raise ValueError("LSTM requires vocab_size")
        return XSSDetectorLSTM(vocab_size)
    elif model_name == "MLP":
        if input_dim is None:
            raise ValueError("MLP requires input_dim")
        return MLPModel(input_dim)
    else:
        raise ValueError("Unknown model name")

# 預測函數
def predict(model, weight_path, input_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    return "Malicious" if pred == 1 else "Normal"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XSS Detection Model Execution Tool",
        epilog="Example usage: python main.py -input payload.txt -model CNN -權重 CNN_model.pth -output result.txt"
    )
    parser.add_argument("-input", required=True, help="Input file containing the text to be analyzed")
    parser.add_argument("-model", required=True, choices=["CNN", "LSTM", "MLP"], help="Select the model to use")
    parser.add_argument("-weight", required=True, help="Specify the corresponding model weight file (.pth)")
    parser.add_argument("-output", required=True, help="Output file to save the result")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weight):
        raise FileNotFoundError(f"Weight file not found: {args.weight}")
    
    # 讀取輸入檔
    with open(args.input, "r") as f:
        input_text = f.read().strip()
        
    if args.model == "CNN":
        input_data = np.random.rand(1, 1, 28, 28)  # CNN requires image format (1, 1, H, W)
    elif args.model == "LSTM":
        vectorizer = TfidfVectorizer()
        input_data = vectorizer.fit_transform([input_text]).toarray()[0]
    elif args.model == "MLP":
        vectorizer = TfidfVectorizer()
        input_data = vectorizer.fit_transform([input_text]).toarray()[0]
    
    model = load_model(args.model, input_dim=len(input_data), vocab_size=5000)
    result = predict(model, args.weight, input_data)
    
    # 輸出結果
    with open(args.output, "w") as f:
        f.write(result)
    
    print(f"Prediction result saved to {args.output}")
