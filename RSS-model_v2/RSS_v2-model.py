import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

# config log
logging.basicConfig(filename='training_bs32_epo100_layer2.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# put device on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# Load data
# data_array = np.load('time_series_data.npy')
# label_array = np.load('labels.npy')
script_path = os.path.abspath(__file__)
data_array = np.load(os.path.join(os.path.dirname(script_path), 'time_series.npy'))
label_array = np.load(os.path.join(os.path.dirname(script_path), 'labels.npy'))

# data normalization
mean = data_array.mean()
std = data_array.std()
data_normalized = (data_array - mean) / std

# Preprocess labels
label_encoder = LabelEncoder()
label_array = label_encoder.fit_transform(label_array)

# Convert to PyTorch tensors
data_tensor = torch.tensor(data_array, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
label_tensor = torch.tensor(label_array, dtype=torch.long)

# Create dataset and split
dataset = TensorDataset(data_tensor, label_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Adjust batch size if necessary
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=4,num_layers = 2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size,num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Training setup
model = LSTMModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate if necessary

# Training loop
num_epochs = 100  # Adjust number of epochs if necessary
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    logging.info(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Testing
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
logging.info(f'Accuracy: {100 * correct / total}%')
