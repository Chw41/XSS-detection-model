XSS dection model (LSTM & MLP)
===

# Enviroment
Systerm:	**Ubuntu 22.04.3 LTS**
CPU:	**CPU E5-2680 V4 @2.40GHz**
GPU:	**NVIDIA RTX A5000**

Python module	Python 3.12.3
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
(chwenv) root@chw:~# python -c "import torch; print(torch.__version__)"
2.4.1+cu121
```

# Create LSTM model
```
vi LSTM.py
```
LSTM.py
```python=
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 只取最後一個時間步驟的輸出
        return output

input_size = 1
hidden_size = 50
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)

print(model)
```
```
(chwenv) root@fc8f6c1b3229:~# python3 LSTM.py
LSTMModel(
  (lstm): LSTM(1, 50, batch_first=True)
  (fc): Linear(in_features=50, out_features=1, bias=True)
)
```

# Create MLP mode
```
vi MLP.py
```
MLP.py
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
