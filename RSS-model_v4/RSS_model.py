import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import seaborn as sns
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# config log
logging.basicConfig(filename='log/V4_training_bs64_epo100_layer2_02.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# put device on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")
def matrix():
    model.eval()
    labels = []
    preds = []
    with torch.no_grad():
        for RSS, label in test_loader:
            RSS, label = RSS.to(device), label.to(device)
            outputs = model(RSS)
            _, predicted = torch.max(outputs.data, 1)

            # Collect the true and predicted labels
            labels.extend(label.cpu().numpy())
            preds.extend(predicted.cpu().numpy())

        # Compute the confusion matrix
        cm = confusion_matrix(labels, preds)
        print(cm)

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm_normalized)

        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm_normalized, annot=False, fmt='.4f', ax=ax, cmap='Blues',  # Turn off automatic annotation
                    xticklabels=classes, yticklabels=classes)

        # Manually add annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j + 0.5, i + 0.5, '{:0.4f}'.format(cm_normalized[i, j]),
                        horizontalalignment='center', verticalalignment='center',
                        color="black" if cm_normalized[i, j] < 0.5 else "white")

        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix -RSS only')
        plt.savefig(f"./plot/RSS_v4_matrix_e.png")

    print('Finished drawing')
# Load data
# data_array = np.load('time_series_data.npy')
# label_array = np.load('labels.npy')
script_path = os.path.abspath(__file__)
data_array = np.load(os.path.join(os.path.dirname(script_path), 'data/v4_RSS_data.npy'))
label_array = np.load(os.path.join(os.path.dirname(script_path), 'data/v4_RSS_labels.npy'))

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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Adjust batch size if necessary
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=4,num_layers = 2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers , batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Training setup
model = LSTMModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate if necessary

classes = ["LIFT", "LR", "PUSH", "PULL"]

# Training loop
num_epochs = 300  # Adjust number of epochs if necessary
train_acc_array = []
test_acc_array = []
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
    logging.info(f'Epoch {epoch+1}, Loss: {loss.item()}')

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
torch.save(model.state_dict(), "parameter/RSS_LSTM2_parameter.pt")
matrix()
#net_clone = LSTMModel()
#net_clone.load_state_dict(torch.load("parameter/RSS_LSTM2_parameter.pt"))

#torch.sigmoid(net_clone.forward(torch.tensor(test_dataset[0:10]).float())).data
