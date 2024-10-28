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
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset  
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

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
arr = np.zeros((len(sentences), 100, 100))
for i in range(len(sentences)):
    image = convert_to_ascii(sentences[i])
    x = np.asarray(image, dtype='float')
    image = cv2.resize(x, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
    image /= 128
    arr[i] = image

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

# Split into train/test data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

train_dataset = XSSDataset(train_data, train_labels)
test_dataset = XSSDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 12 * 12, 256)  # 修改展平後的大小
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(x.size(0), -1)  # 自動計算展平後的大小
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# Initialize model, loss function and optimizer
model = CNNModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction
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
confusion = confusion_matrix(test_labels, predictions)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("Confusion Matrix: \n", confusion)


```
![image](https://github.com/user-attachments/assets/946b2455-e38d-48a3-b0e8-aa88e6445995)

![image](https://github.com/user-attachments/assets/068648a5-1475-45cf-83fe-d065841a415d)

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

# Create MLP mode
```
vi MLP.py
```
[MLP.py](https://github.com/Chw41/XSS-dection-model/blob/main/MLP.py)
```python=
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = 100  # 假設有100個輸入特徵
mlp_model = MLPModel(input_size)

print(mlp_model)
```
```
(chwenv) root@fc8f6c1b3229:~# python3 MLP.py
MLPModel(
  (fc1): Linear(in_features=100, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=1, bias=True)
)
```
