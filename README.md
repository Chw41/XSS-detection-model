XSS detection model (LSTM & MLP)
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

# Create LSTM model
```
pip install torch torchvision torchaudio
pip install pandas
pip install numpy
pip install opencv-python
pip install scikit-learn
pip install matplotlib
```

```
vi LSTM.ipynb
```
[LSTM.ipynb](https://github.com/Chw41/XSS-dection-model/blob/main/LSTM.ipynb)
```python=

```
![image](https://github.com/user-attachments/assets/946b2455-e38d-48a3-b0e8-aa88e6445995)

![image](https://github.com/user-attachments/assets/068648a5-1475-45cf-83fe-d065841a415d)


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
