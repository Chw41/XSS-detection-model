XSS detection model (CNN & LSTM & MLP)
===

# Enviroment
Systerm:	**Ubuntu 22.04.3 LTS**\
CPU:	**CPU E5-2680 V4 @2.40GHz**\
GPU:	**NVIDIA RTX A5000**

Python module	Python 3.12.3\
pytorch=2.4.1+cu121

# Env setup
## - Install Python
```
sudo apt update
sudo apt install python3-pip
apt install python3.12-venv
```
## - Create virtual environment
```
root@chw:~# python3 -m venv chwenv
root@chw:~# source chwenv/bin/activate
```
## - Install PyTorch & Check
```
(chwenv) root@chw:~# pip3 install torch torchvision
(chwenv) root@chw:~# python3 -c "import torch; print(torch.__version__)"
2.5.0+cu124
```

# Create CNN model
```
pip install torch torchvision torchaudio
pip install pandas
pip install numpy
pip install opencv-python
pip install scikit-learn
pip install matplotlib==3.8.0
```

```
vi CNN.ipynb
```
[CNN.ipynb](https://github.com/Chw41/XSS-dection-model/blob/main/CNN.ipynb)
```python=
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset  
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load Dataset
df = pd.read_csv('dataset/XSS_dataset.csv', encoding='utf-8-sig')
df = df[df.columns[-2:]]  # Only get sentence and labels

# Get Sentences data from data frame
sentences = df['Sentence'].values
labels = df['Label'].values

# Convert to ASCII
def convert_to_ascii(sentence):
    sentence_ascii = []
    for char in sentence:
        if ord(char) < 8222:
            if ord(char) == 8217:  # ’
                sentence_ascii.append(134)
            elif ord(char) == 8221:  # ”
                sentence_ascii.append(129)
            elif ord(char) == 8220:  # “
                sentence_ascii.append(130)
            elif ord(char) == 8216:  # ‘
                sentence_ascii.append(131)
            elif ord(char) == 8211:  # –
                sentence_ascii.append(133)
            if ord(char) <= 128:
                sentence_ascii.append(ord(char))
    zer = np.zeros((10000,))
    for i in range(len(sentence_ascii)):
        zer[i] = sentence_ascii[i]
    zer.shape = (100, 100)
    return zer

# Prepare Data
arr = np.zeros((len(sentences), 10, 10))
for i in range(len(sentences)):
    image = convert_to_ascii(sentences[i])
    x = np.asarray(image, dtype='float')
    image = cv2.resize(x, dsize=(10, 10), interpolation=cv2.INTER_CUBIC)
    image /= 128
    arr[i] = image

# Show the first two images in the data array
for i in range(2):
    plt.imshow(arr[i], cmap='gray')
    plt.title(f'Image {i + 1}')
    plt.show()

# Reshape data for input to CNN
data = arr.reshape(arr.shape[0], 1, 100, 100)

# Create PyTorch Dataset
class XSSDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Split data: train 70%, verify 20%, test 10%
train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=0.3, random_state=42)
verify_data, test_data, verify_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=1/3, random_state=42)

train_dataset = XSSDataset(train_data, train_labels)
verify_dataset = XSSDataset(verify_data, verify_labels)
test_dataset = XSSDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
verify_loader = DataLoader(verify_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define CNN Model
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
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# Train and test with different learning rates
learning_rates = [0.001, 0.002, 0.01, 0.02, 0.05]
num_epochs = 20
all_loss_train = []
all_loss_verify = []

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    model = CNNModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_train = []
    loss_verify = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_train.append(epoch_loss / len(train_loader))

        # Validation loss
        model.eval()
        verify_loss = 0
        with torch.no_grad():
            for inputs, labels in verify_loader:
                outputs = model(inputs.float())
                loss = criterion(outputs.squeeze(), labels.float())
                verify_loss += loss.item()
        loss_verify.append(verify_loss / len(verify_loader))

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss_train[-1]:.4f}, Verify Loss: {loss_verify[-1]:.4f}")

    all_loss_train.append(loss_train)
    all_loss_verify.append(loss_verify)

# Plot training and verify loss for each learning rate
for i, lr in enumerate(learning_rates):
    plt.plot(all_loss_train[i], label=f'Train Loss (lr={lr})')
    plt.plot(all_loss_verify[i], label=f'Verify Loss (lr={lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss Curves for learning rate = {lr}')
    plt.show()

# Prediction on test data
model.eval()
predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.float())
        predictions.extend(outputs.squeeze().round().numpy())

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)
confusion = confusion_matrix(test_labels, predictions)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("Confusion Matrix: \n", confusion)

```

```
# Prepare Data
顯示結果圖片 2,3 張

dataset 分類
train 7 verfy 2 test 1

dataloader 最少要8 個 （補充）

optimizer 需要有
0.001, 0.002, 0.01, 0.02, 0.05

計算 f1 score

train loop 最少 20 次
化成 loss 折線圖
train losss test loss
```

```

Training with learning rate: 0.001
Epoch [1/20], Training Loss: 0.4829, Verify Loss: 0.4641
Epoch [2/20], Training Loss: 0.3413, Verify Loss: 0.3003
Epoch [3/20], Training Loss: 0.1386, Verify Loss: 0.0985
Epoch [4/20], Training Loss: 0.0929, Verify Loss: 0.0886
Epoch [5/20], Training Loss: 0.0709, Verify Loss: 0.0990
Epoch [6/20], Training Loss: 0.0662, Verify Loss: 0.0760
Epoch [7/20], Training Loss: 0.0615, Verify Loss: 0.0684
Epoch [8/20], Training Loss: 0.0559, Verify Loss: 0.0629
Epoch [9/20], Training Loss: 0.0465, Verify Loss: 0.0587
Epoch [10/20], Training Loss: 0.0437, Verify Loss: 0.0663
Epoch [11/20], Training Loss: 0.0388, Verify Loss: 0.0564
Epoch [12/20], Training Loss: 0.0382, Verify Loss: 0.0543
Epoch [13/20], Training Loss: 0.0327, Verify Loss: 0.0529
Epoch [14/20], Training Loss: 0.0310, Verify Loss: 0.0435
Epoch [15/20], Training Loss: 0.0324, Verify Loss: 0.0414
Epoch [16/20], Training Loss: 0.0252, Verify Loss: 0.0371
Epoch [17/20], Training Loss: 0.0222, Verify Loss: 0.0484
Epoch [18/20], Training Loss: 0.0206, Verify Loss: 0.0323
Epoch [19/20], Training Loss: 0.0219, Verify Loss: 0.0583
Epoch [20/20], Training Loss: 0.0169, Verify Loss: 0.0313

Training with learning rate: 0.002
Epoch [1/20], Training Loss: 0.4896, Verify Loss: 0.3761
Epoch [2/20], Training Loss: 0.2332, Verify Loss: 0.1071
Epoch [3/20], Training Loss: 0.0995, Verify Loss: 0.1006
Epoch [4/20], Training Loss: 0.0790, Verify Loss: 0.0783
Epoch [5/20], Training Loss: 0.0687, Verify Loss: 0.0890
Epoch [6/20], Training Loss: 0.0641, Verify Loss: 0.0725
Epoch [7/20], Training Loss: 0.0574, Verify Loss: 0.0666
Epoch [8/20], Training Loss: 0.0574, Verify Loss: 0.0560
Epoch [9/20], Training Loss: 0.0494, Verify Loss: 0.0648
Epoch [10/20], Training Loss: 0.0395, Verify Loss: 0.0557
Epoch [11/20], Training Loss: 0.0349, Verify Loss: 0.0525
Epoch [12/20], Training Loss: 0.0367, Verify Loss: 0.0635
Epoch [13/20], Training Loss: 0.0294, Verify Loss: 0.0557
Epoch [14/20], Training Loss: 0.0286, Verify Loss: 0.0550
Epoch [15/20], Training Loss: 0.0268, Verify Loss: 0.0384
Epoch [16/20], Training Loss: 0.0326, Verify Loss: 0.0719
Epoch [17/20], Training Loss: 0.0227, Verify Loss: 0.0567
Epoch [18/20], Training Loss: 0.0206, Verify Loss: 0.0482
Epoch [19/20], Training Loss: 0.0172, Verify Loss: 0.0389
Epoch [20/20], Training Loss: 0.0134, Verify Loss: 0.0624

Training with learning rate: 0.01
Epoch [1/20], Training Loss: 53.1779, Verify Loss: 54.4382
Epoch [2/20], Training Loss: 53.7909, Verify Loss: 54.4382
Epoch [3/20], Training Loss: 53.7967, Verify Loss: 54.4382
Epoch [4/20], Training Loss: 53.7870, Verify Loss: 54.4382
Epoch [5/20], Training Loss: 53.7755, Verify Loss: 54.4382
Epoch [6/20], Training Loss: 53.7890, Verify Loss: 54.4382
Epoch [7/20], Training Loss: 53.7870, Verify Loss: 54.4382
Epoch [8/20], Training Loss: 53.7909, Verify Loss: 54.4382
Epoch [9/20], Training Loss: 53.7967, Verify Loss: 54.4382
Epoch [10/20], Training Loss: 53.7909, Verify Loss: 54.4382
Epoch [11/20], Training Loss: 53.7928, Verify Loss: 54.4382
Epoch [12/20], Training Loss: 53.8005, Verify Loss: 54.4382
Epoch [13/20], Training Loss: 53.7812, Verify Loss: 54.4382
Epoch [14/20], Training Loss: 53.7870, Verify Loss: 54.4382
Epoch [15/20], Training Loss: 53.7909, Verify Loss: 54.4382
Epoch [16/20], Training Loss: 53.8005, Verify Loss: 54.4382
Epoch [17/20], Training Loss: 53.7793, Verify Loss: 54.4382
Epoch [18/20], Training Loss: 53.8005, Verify Loss: 54.4382
Epoch [19/20], Training Loss: 53.7948, Verify Loss: 54.4382
Epoch [20/20], Training Loss: 53.7832, Verify Loss: 54.4382

Training with learning rate: 0.02
Epoch [1/20], Training Loss: 45.6042, Verify Loss: 45.5618
Epoch [2/20], Training Loss: 46.2245, Verify Loss: 45.5618
Epoch [3/20], Training Loss: 46.1937, Verify Loss: 45.5618

```

Ref: https://github.com/harikrizz77/XSS-attack-detection-using-LSTM/blob/main/code.ipynb

# Create LSTM mode
```
vi LSTM.ipynb
```
[LSTM.ipynb](https://github.com/Chw41/XSS-dection-model/blob/main/LSTM.ipynb)
```python=
# Import necessary libraries
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
from typing import List, Tuple

# Preprocessing class for handling text data
class TextPreprocessor:
    def __init__(self, max_len: int = 100):
        self.max_len = max_len
        self.vocab = {'<PAD>': 0, '<UNK>': 1}  # Initial vocabulary with padding and unknown tokens
        self.vocab_size = 2
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize HTML text into meaningful components."""
        # Separate HTML tags and attributes
        text = re.sub(r'([<>/="])', r' \1 ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower().split()
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary with common tokens"""
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        
        # Add words meeting the minimum frequency threshold
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.vocab:
                self.vocab[word] = self.vocab_size
                self.vocab_size += 1
    
    def encode_text(self, text: str) -> List[int]:
        """Convert text to integer sequence"""
        tokens = self.tokenize(text)
        # Truncate or pad to specified length
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens.extend(['<PAD>'] * (self.max_len - len(tokens)))
        
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

# Custom Dataset class for XSS data
class XSSDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.encodings = [self.preprocessor.encode_text(text) for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.encodings[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float))

# LSTM model class for XSS detection
class XSSDetectorLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 50, 
                 hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use the last time step's hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.dropout(hidden)
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

# Detector class to handle training and prediction
class XSSDetector:
    def __init__(self, max_len: int = 100, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.max_len = max_len
        self.preprocessor = TextPreprocessor(max_len)
        self.model = None
    
    def train(self, texts: List[str], labels: List[int], 
              epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001,
              val_split: float = 0.1):
        
        # Build vocabulary
        self.preprocessor.build_vocab(texts)
        
        # Create dataset
        dataset = XSSDataset(texts, labels, self.preprocessor)
        
        # Split into training and validation sets
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        self.model = XSSDetectorLSTM(
            vocab_size=self.preprocessor.vocab_size,
            embedding_dim=50
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_sequences, batch_labels in train_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_sequences).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = (outputs >= 0.5).float()
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_sequences, batch_labels in val_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_sequences).squeeze()
                    val_loss += criterion(outputs, batch_labels).item()
                    
                    predictions = (outputs >= 0.5).float()
                    val_correct += (predictions == batch_labels).sum().item()
                    val_total += len(batch_labels)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Training Loss: {total_loss/len(train_loader):.4f}')
            print(f'Training Accuracy: {100*correct/total:.2f}%')
            print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
            print(f'Validation Accuracy: {100*val_correct/val_total:.2f}%\n')
    
    def predict(self, text: str) -> float:
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        
        self.model.eval()
        sequence = self.preprocessor.encode_text(text)
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.model(sequence_tensor)
            return output.item()

# Function to demonstrate the detector with example inputs
def demo_detector():
    # Load dataset
    data = pd.read_csv('dataset/XSS_dataset.csv')
    texts = data['Sentence'].tolist()
    labels = data['Label'].tolist()
    
    # Initialize detector
    detector = XSSDetector(max_len=100)
    
    # Train model
    detector.train(
        texts=texts,
        labels=labels,
        epochs=20,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Test with some example cases
    test_cases = [
        "<script>alert(1)</script>",
        "<img src='x' onerror='alert(1)'>",
        "<a href='/wiki/Portal:Philosophy'>Philosophy portal</a>",
        "<span class='reference-text'>Normal text</span>"
    ]
    
    print("\nTesting some examples:")
    for test_input in test_cases:
        risk_score = detector.predict(test_input)
        print(f"Risk score for '{test_input}': {risk_score:.3f}")

# Run demonstration
if __name__ == "__main__":
    demo_detector()
```
![image](https://github.com/user-attachments/assets/a8e32a1e-a35a-4b54-bc51-4db8164ca597)

![image](https://github.com/user-attachments/assets/e55cfe09-9c1c-4ed2-a029-c5c6ae785cd8)

# Create MLP mode
```
pip3 install gensim nltk
```

```
vi MLP.ipynb
```
[MLP.ipynb](https://github.com/Chw41/XSS-dection-model/blob/main/MLP.ipynb)
```python=

```

Ref: https://github.com/antonmedv/fx](https://github.com/obarrera/ML-XSS-Detection/blob/main/XSS-Doc2Vec-ML-Classifier.ipynb
