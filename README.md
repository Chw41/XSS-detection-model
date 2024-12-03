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
    print(f"Training with learning rate: {lr}")

    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Lists to store losses
    self.training_losses = []
    self.validation_losses = []
            

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
plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(self.training_losses) + 1), self.training_losses, label='Training Loss')
    plt.plot(range(1, len(self.validation_losses) + 1), self.validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Over Time (Learning Rate: {lr})')
    plt.legend()
    plt.grid(True)
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
![image](https://github.com/user-attachments/assets/71654f93-7b72-447e-95d1-9f73fa3a2ec4)


Ref: https://github.com/harikrizz77/XSS-attack-detection-using-LSTM/blob/main/code.ipynb

# Create LSTM model
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
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os

class TextPreprocessor:
    def __init__(self, max_len: int = 100):
        self.max_len = max_len
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.vocab_size = 2
    
    def tokenize(self, text: str) -> List[str]:
        # Convert input to string and clean it
        text = str(text)
        text = re.sub(r'([<>/="])', r' \1 ', text)
        text = ' '.join(text.split())
        return text.lower().split()
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        counter = Counter()
        for text in texts:
            # Ensure text is string
            text = str(text)
            tokens = self.tokenize(text)
            counter.update(tokens)
        
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.vocab:
                self.vocab[word] = self.vocab_size
                self.vocab_size += 1
    
    def encode_text(self, text: str) -> List[int]:
        # Ensure text is string
        text = str(text)
        tokens = self.tokenize(text)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens.extend(['<PAD>'] * (self.max_len - len(tokens)))
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]


class XSSDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], preprocessor: TextPreprocessor):
        # Convert all texts to strings
        self.texts = [str(text) for text in texts]
        self.preprocessor = preprocessor
        self.encodings = [self.preprocessor.encode_text(text) for text in self.texts]
        self.labels = [int(label) for label in labels]  # Convert labels to int

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.encodings[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float))

def load_and_clean_data(file_path: str) -> Tuple[List[str], List[int]]:
    """Load and clean the dataset, ensuring proper data types."""
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Convert texts to strings and clean them
        texts = [str(text).strip() for text in data['Sentence']]
        
        # Convert labels to integers
        labels = [int(label) for label in data['Label']]
        
        # Basic validation
        assert len(texts) == len(labels), "Number of texts and labels must match"
        assert all(isinstance(text, str) for text in texts), "All texts must be strings"
        assert all(isinstance(label, int) and label in [0, 1] for label in labels), "Labels must be binary (0 or 1)"
        
        print(f"Loaded {len(texts)} samples successfully")
        
        # Print some basic statistics
        print(f"Number of positive samples: {sum(labels)}")
        print(f"Number of negative samples: {len(labels) - sum(labels)}")
        
        return texts, labels
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

class XSSDetectorLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 50, 
                 hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
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
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.dropout(hidden)
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

class XSSDetector:
    def __init__(self, max_len: int = 100, device: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        print(f"Using device: {self.device}")
        
        self.max_len = max_len
        self.preprocessor = TextPreprocessor(max_len)
        self.model = None
        self.results = {}
    
    def train(self, texts: List[str], labels: List[int], 
              epochs: int = 20, batch_size: int = 16,  
              learning_rates: List[float] = [0.001, 0.002, 0.01, 0.02, 0.05]):
        
        try:
            texts = [str(text) for text in texts]
            labels = torch.tensor(labels, dtype=torch.float)
            
            self.preprocessor.build_vocab(texts)
            dataset = XSSDataset(texts, labels.numpy(), self.preprocessor)
            
            # Split dataset
            train_size = int(0.7 * len(dataset))
            val_size = int(0.2 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )
            
            print(f"\nDataset splits:")
            print(f"Training: {train_size} samples")
            print(f"Validation: {val_size} samples")
            print(f"Test: {test_size} samples")
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Train with different learning rates
            for lr in learning_rates:
                print(f"\nTraining with learning rate: {lr}")
                
                self.model = XSSDetectorLSTM(
                    vocab_size=self.preprocessor.vocab_size,
                    embedding_dim=50,
                    dropout=0.3  # Increased dropout for regularization
                ).to(self.device)
                
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
                criterion = nn.BCELoss()
                
                # Learning rate scheduler for better convergence
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=3, verbose=True
                )
                
                train_losses = []
                val_losses = []
                
                best_val_loss = float('inf')
                patience_counter = 0
                
                for epoch in range(epochs):
                    # Training
                    self.model.train()
                    total_loss = 0
                    train_correct = 0
                    train_total = 0
                    
                    for batch_sequences, batch_labels in train_loader:
                        batch_sequences = batch_sequences.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = self.model(batch_sequences).squeeze()
                        loss = criterion(outputs, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        predictions = (outputs >= 0.5).float()
                        train_correct += (predictions == batch_labels).sum().item()
                        train_total += len(batch_labels)
                    
                    avg_train_loss = total_loss / len(train_loader)
                    train_losses.append(avg_train_loss)
                    
                    # Validation
                    self.model.eval()
                    val_loss = 0
                    val_correct = 0
                    val_total = 0
                    val_predictions = []
                    val_true_labels = []
                    
                    with torch.no_grad():
                        for batch_sequences, batch_labels in val_loader:
                            batch_sequences = batch_sequences.to(self.device)
                            batch_labels = batch_labels.to(self.device)
                            
                            outputs = self.model(batch_sequences).squeeze()
                            batch_val_loss = criterion(outputs, batch_labels).item()
                            val_loss += batch_val_loss
                            
                            predictions = (outputs >= 0.5).float()
                            val_correct += (predictions == batch_labels).sum().item()
                            val_total += len(batch_labels)
                            
                            val_predictions.extend(predictions.cpu().numpy())
                            val_true_labels.extend(batch_labels.cpu().numpy())
                    
                    avg_val_loss = val_loss / len(val_loader)
                    val_losses.append(avg_val_loss)
                    
                    # Learning rate scheduler step
                    scheduler.step(avg_val_loss)
                    
                    # Early stopping
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        # Save the best model
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'preprocessor': self.preprocessor
                        }, f'best_model_lr_{lr}.pth')
                    else:
                        patience_counter += 1
                    
                    if patience_counter > 5:
                        print("Early stopping triggered")
                        break
                    
                    if (epoch + 1) % 5 == 0:
                        print(f'Epoch {epoch+1}/{epochs}:')
                        print(f'Training Loss: {avg_train_loss:.4f}')
                        print(f'Training Accuracy: {100*train_correct/train_total:.2f}%')
                        print(f'Validation Loss: {avg_val_loss:.4f}')
                        print(f'Validation Accuracy: {100*val_correct/val_total:.2f}%')
                
                # Final evaluation
                val_f1 = f1_score(val_true_labels, val_predictions)
                
                # Store results
                self.results[lr] = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'final_train_loss': avg_train_loss,
                    'final_val_loss': avg_val_loss,
                    'train_accuracy': 100 * train_correct / train_total,
                    'val_accuracy': 100 * val_correct / val_total,
                    'f1_score': val_f1
                }
                
                # Plot training curves with consistent scaling
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
                plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Training and Validation Loss (Learning Rate: {lr})')
                plt.ylim(0, 1)  # Consistent y-axis
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'loss_plot_lr_{lr}.png')
                plt.close()
            
            # Print final results
            print("\nFinal Results for All Learning Rates:")
            print("\nLR      Train Loss  Val Loss    Train Acc   Val Acc    F1 Score")
            print("-" * 65)
            for lr in learning_rates:
                r = self.results[lr]
                print(f"{lr:.3f}  {r['final_train_loss']:.4f}     {r['final_val_loss']:.4f}     "
                      f"{r['train_accuracy']:.2f}%     {r['val_accuracy']:.2f}%     {r['f1_score']:.4f}")
        
        except Exception as e:
            print(f"Training error: {e}")
    
    def predict(self, texts: List[str], model_path: str):
        """
        Load a saved model and predict on new texts
        
        Args:
            texts (List[str]): Input texts to predict
            model_path (str): Path to saved model checkpoint
        
        Returns:
            List[int]: Predictions (0 or 1)
        """
        # Load the saved model and preprocessor
        checkpoint = torch.load(model_path)
        saved_preprocessor = checkpoint['preprocessor']
        
        # Ensure texts are processed the same way
        encoded_texts = [saved_preprocessor.encode_text(str(text)) for text in texts]
        
        # Prepare model
        model = XSSDetectorLSTM(
            vocab_size=saved_preprocessor.vocab_size,
            embedding_dim=50
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Predict
        with torch.no_grad():
            text_tensors = torch.tensor(encoded_texts, dtype=torch.long)
            outputs = model(text_tensors).squeeze()
            predictions = (outputs >= 0.5).int().numpy()
        
        return predictions.tolist()


# Pre-execution environment check
def check_environment():
    print("\n--- Environment Check ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    try:
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    except:
        print("No CUDA device currently selected")

def demo_detector(dataset_path='Training Dataset/final_dataset.csv'):
    print("\n--- XSS Detection Model Demonstration ---")
    
    try:
        # Load and clean data
        texts, labels = load_and_clean_data(dataset_path)
        
        # Initialize and train detector
        detector = XSSDetector(max_len=100)
        detector.train(
            texts=texts,
            labels=labels,
            epochs=20,
            batch_size=16,
            learning_rates=[0.001, 0.002, 0.01, 0.02, 0.05]
        )
    
    except Exception as e:
        print(f"Demonstration failed: {e}")
        print("Possible issues:")
        print("1. Ensure correct dataset path")
        print("2. Check dataset format")
        print("3. Verify required libraries are installed")


if __name__ == "__main__":
    check_environment()
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
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

class XSSDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs):
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_epoch_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        
        train_epoch_loss /= len(train_loader)
        train_losses.append(train_epoch_loss)
        
        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_epoch_loss += loss.item()
            
            val_epoch_loss /= len(val_loader)
            val_losses.append(val_epoch_loss)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss = {train_epoch_loss:.4f}, Val Loss = {val_epoch_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_metrics(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    return {
        'F1 Score': f1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

def main():
    # Load dataset
    dataset_path = '/Training Dataset/final_dataset.csv'
    
    # Read CSV and handle NaN values
    df = pd.read_csv(dataset_path)
    
    # Remove rows with NaN values in 'Sentence' or 'Label' columns
    df = df.dropna(subset=['Sentence', 'Label'])
    
    # Convert 'Sentence' to string type and replace any remaining NaNs
    df['Sentence'] = df['Sentence'].astype(str).fillna('')
    
    # Print dataset info
    print("Dataset shape after cleaning:", df.shape)
    print("\nSample of cleaned dataset:")
    print(df.head())
    
    # Text Vectorization
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['Sentence']).toarray()
    y = df['Label'].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    
    # Create data loaders
    train_dataset = XSSDataset(X_train, y_train)
    val_dataset = XSSDataset(X_val, y_val)
    test_dataset = XSSDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Print first 3 samples
    print("\nFirst 3 Training Samples:")
    for i, (features, label) in enumerate(train_loader):
        if i < 1:
            print("Features shape:", features[:3].shape)
            print("Labels:", label[:3])
        break
    
    # Learning rates to experiment
    learning_rates = [0.001, 0.002, 0.01, 0.02, 0.05]
    epochs = 20
    results = {}
    
    # Create a figure with 5 subplots, one for each learning rate
    plt.figure(figsize=(20, 15))
    
    for lr in learning_rates:
        print(f"\n--- Learning Rate: {lr} ---")
        
        # Reset model and optimizer for each learning rate
        model = MLPModel(X_train.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        train_losses, val_losses = train_and_evaluate(
            model, train_loader, val_loader, optimizer, criterion, epochs
        )
        
        # Plot losses for this learning rate in a separate subplot
        plt.subplot(2, 3, learning_rates.index(lr) + 1)
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o')
        plt.title(f'Loss Curves - Learning Rate: {lr}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, epochs + 1))
        
        # Evaluate metrics
        metrics = evaluate_metrics(model, test_loader)
        results[lr] = metrics
        
        print("Metrics:", metrics)
    
    plt.tight_layout()
    plt.savefig('learning_rate_losses.png')
    plt.close()
    
    # Print comprehensive results
    print("\n--- Comprehensive Results ---")
    for lr, metrics in results.items():
        print(f"\nLearning Rate: {lr}")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
```

Ref: https://github.com/antonmedv/fx](https://github.com/obarrera/ML-XSS-Detection/blob/main/XSS-Doc2Vec-ML-Classifier.ipynb
