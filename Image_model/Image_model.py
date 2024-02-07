import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

script_path = os.path.abspath(__file__)
data_array = np.load(os.path.join(os.path.dirname(script_path), 'image_data.npy'))
label_array = np.load(os.path.join(os.path.dirname(script_path), 'label_data.npy'))

data_array = data_array/255.0

label_encoder = LabelEncoder()
label_array = label_encoder.fit_transform(label_array)

#transoform array to tensor
data_tensor = torch.tensor(data_array, dtype=torch.float32)
label_tensor = torch.tensor(label_array, dtype=torch.long)


dataset = TensorDataset(data_tensor, label_tensor)

#Split train/test dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride = 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 129, 3, stride = 3)
        self.fc1 = nn.Linear(41796, 128)
        self.fc2 = nn.Linear(128, 4)  # num_classes 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 41796)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CNN()
model.to(device) 
# Loss Function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# train
num_epochs = 100
for epoch in range(num_epochs):
    train_loop = tqdm(train_loader, leave=True, miniters=1)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        train_loop.set_description(f'Epoch {epoch+1}/{num_epochs}')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loop.update()
    train_loop.set_postfix(loss=loss.item())
print('Finished Training')

# test
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        images, labels = inputs.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # print(outputs)
        # print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
