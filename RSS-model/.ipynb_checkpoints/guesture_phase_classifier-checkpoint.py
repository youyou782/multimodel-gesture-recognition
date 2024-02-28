import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split

def load_bin_file(filepath):
    """Load a single .bin file as a PyTorch tensor."""
    with open(filepath, 'rb') as f:
        binary_data = f.read()
    numpy_array = np.frombuffer(binary_data, dtype=np.float32)
    tensor = torch.from_numpy(numpy_array).to(torch.float32)
    return tensor

def load_data_from_folder(folder_path):
    """Load all .bin files from the specified folder into tensors."""
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.bin')]
    data = [load_bin_file(f) for f in files]
    return data

# Path to your folder containing .bin files
folder_path = './Phase/Mix'

# Load all data
data = load_data_from_folder(folder_path)

# Split the data into training and testing sets(list): 306 : 54
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

sequence_len = 18000
input_len = 1
hidden_size = 128
num_layers = 2
num_classes = 4
num_epoch = 10
learning_rate = 0.01
batch_size = 1

class LSTM(nn.Module):
    def __init__(self,input_len, hidden_size, num_classes, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_len = input_len
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_len, self.hidden_size, self.num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        out, _ = self.lstm(X, (hidden_states, cell_states))
        out = self.output_layer(out[:, -1, :])
        return out

model = LSTM(input_len, hidden_size, num_classes, num_layers)
print(model)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_model(train_data, test_data, model, loss_func, optimizer, num_epochs):
    total_steps = len(train_data)

    for epoch  in range(num_epochs):
        for batch, labels in train_data:
            # Forward pass
            outputs = model(batch)
            loss = loss_func(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch + 1}/{total_steps}], Loss: {loss.item():.4f}')


# visualization loss
plt.plot(epoch_list, val_loss_list)
plt.xlabel("# of epochs")
plt.ylabel("Loss")
plt.title("LSTM: Loss vs # epochs")
plt.show()

# visualization accuracy
plt.plot(epoch_list, val_accuracy_list, color="red")
plt.xlabel("# of epochs")
plt.ylabel("Accuracy")
plt.title("LSTM: Accuracy vs # epochs")
# plt.savefig('graph.png')
plt.show()
