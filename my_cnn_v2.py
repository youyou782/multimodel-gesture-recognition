import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")


class MultiModalDataset(Dataset):
    def __init__(self, image, phase, rss, label, augment=False):
        self.image = image
        self.phase = phase
        self.rss = rss
        self.label = label
        self.augment = augment
        self.image_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(size=(image.shape[2], image.shape[3]), scale=(0.8, 1.0))
        ])

    def __len__(self):
        # Expanded dataset size
        if self.augment:
            return len(self.label) * 10
        else:
            return len(self.label)

    def __getitem__(self, idx):
        # Adjust idx for data expansion
        idx = idx % len(self.label)

        image = self.image[idx]
        phase = self.phase[idx]
        rss = self.rss[idx]
        label = self.label[idx]

        # Apply image augmentation
        if self.augment:
            image = self.image_transforms(image)

            # Apply jittering augmentation for RSS and phase
            noise_level = 0.05
            phase += noise_level * torch.randn_like(phase)
            rss += noise_level * torch.randn_like(rss)

        return image, phase, rss, label


def load_dataset():
    script_path = os.path.abspath(__file__)
    image_path = os.path.join(os.path.dirname(script_path), "image_data_normalized.npy")
    phase_path = os.path.join(os.path.dirname(script_path), "v4_Phase_data.npy")
    rss_path = os.path.join(os.path.dirname(script_path), "v4_RSS_data.npy")
    label_path = os.path.join(os.path.dirname(script_path), "image_label.npy")

    label = np.load(label_path)
    n_sample = label.shape[0]
    indsh = (np.arange(n_sample))
    np.random.shuffle(indsh)
    labels = label.copy()[indsh]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = torch.tensor(labels, dtype=torch.long)
    print("label.size: ", labels.size())

    labels_train = labels[:int(0.8 * n_sample)]
    labels_test = labels[int(0.8 * n_sample):]


    image = np.load(image_path)
    images = image.copy()[indsh, :]
    images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    print("image.size: ", images.size())

    images_train = images[:int(0.8 * n_sample), :]
    images_test = images[int(0.8 * n_sample):, :]

    phase = np.load(phase_path)
    phases = phase.copy()[indsh, :]
    phases = torch.tensor(phases, dtype=torch.float32).unsqueeze(-1)
    phases = phases.permute(0, 2, 1)
    print("phase.size: ", phases.size())
    phases_train = phases[:int(0.8 * n_sample), :]
    phases_test = phases[int(0.8 * n_sample):, :]

    rss = np.load(rss_path)
    rsss = rss.copy()[indsh, :]
    rsss = torch.tensor(rsss, dtype=torch.float32).unsqueeze(-1)
    rsss = rsss.permute(0, 2, 1)
    print("rss.size: ", rsss.size())

    rsss_train = rsss[:int(0.8 * n_sample), :]
    rsss_test = rsss[int(0.8 * n_sample):, :]

    train_dataset = MultiModalDataset(images_train, phases_train, rsss_train, labels_train, augment=False)
    test_dataset = MultiModalDataset(images_test, phases_test, rsss_test, labels_test, augment=False)

    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    return train_loader, test_loader


script_path = os.path.abspath(__file__)
train_loader, test_loader = load_dataset()


# Modify VGG16
class ModifiedVGG16(nn.Module):
    def __init__(self, feature_size):
        super(ModifiedVGG16, self).__init__()
        self.feature_size = feature_size
        original_vgg16 = models.vgg16(pretrained=True)
        original_vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.features = original_vgg16.features

        self.intermediate_classifier = nn.Sequential(
            *list(original_vgg16.classifier.children())[:-1],  # Exclude the last layer
            nn.Linear(4096, self.feature_size),  # Feature layer
        )
        self.final_classifier = nn.Linear(self.feature_size, 4)
        self.bn = nn.BatchNorm1d(self.feature_size)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        features = self.intermediate_classifier(x)
        # features = self.bn(features)
        features = self.final_classifier(features)
        return features, features


############################################################################################
class CNN1DModel(nn.Module):
    def __init__(self, input_size=1, output_size=4):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.cnn1 = nn.Conv1d(in_channels=self.input_size,
                              out_channels=8, kernel_size=10, stride=3, padding=2)
        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=17, stride=2, padding=0)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.output_size)

    def forward(self, input_seq):
        features = F.relu(self.cnn1(input_seq))  # torch.Size([64, 16, 32])
        features = features.reshape(features.shape[0], -1)

        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        features = self.fc3(features)
        return features, features


############################################################################################
def pretrainer(model,
               train_loader=train_loader,
               test_loader=test_loader,
               num_epoch_pretrain=100,
               l_r=1e-4,
               input_index=1):
    optimizer = optim.Adam(model.parameters(), lr=l_r)
    criterion = nn.CrossEntropyLoss()
    train_loss = []
    test_loss = []
    for epoch in range(num_epoch_pretrain):
        model.train()
        for images, phase, rss, labels in train_loader:
            if input_index == 1:
                inputs = images.to(device)
            elif input_index == 2:
                inputs = phase.to(device)
            else:
                inputs = rss.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)  # Get outputs and ignore features during training
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())

        model.eval()
        correct = 0
        total = 0
        y_pred = []
        y_true = []
        with torch.no_grad():
            for images, phase, rss, labels in test_loader:
                if input_index == 1:
                    inputs = images.to(device)
                elif input_index == 2:
                    inputs = phase.to(device)
                else:
                    inputs = rss.to(device)
                labels = labels.to(device)

                outputs, _ = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
            test_loss.append(loss.item())
        if (epoch % 10 == 0):
            print(f'\n Accuracy on the test set: {100 * correct / total}%')
    cf_matrix = confusion_matrix(y_true, y_pred)
    print('\n***********')
    print(cf_matrix)
    print('***********\n')
    return train_loss, test_loss


#############################################################################################
feature_size = 4
#############################################################################################
# train vgg16
print('################ image ##################\n')
model_image = ModifiedVGG16(feature_size=feature_size).to(device)
train_loss, test_loss = pretrainer(model=model_image,
                                   num_epoch_pretrain=2,
                                   input_index=1)
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig(os.path.join(os.path.dirname(script_path), "train_image.png"))
#############################################################################################
# Train LSTM models
print('################ phase ##################\n')
model_time_series1 = CNN1DModel(output_size=feature_size).to(device)

model_time_series2 = CNN1DModel(output_size=feature_size).to(device)

train_loss, test_loss = pretrainer(model=model_time_series1,
                                   num_epoch_pretrain=2,
                                   l_r=1e-4,
                                   input_index=2)

plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig(os.path.join(os.path.dirname(script_path), "train_phase.png"))
print('################ rss ##################')
train_loss, test_loss = pretrainer(model=model_time_series2,
                                   num_epoch_pretrain=2,
                                   l_r=1e-4,
                                   input_index=3)
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig(os.path.join(os.path.dirname(script_path), "train_rss.png"))


########################################################################################
# define multi-modal fusion
class MultiModelAttention(nn.Module):
    def __init__(self, features_size,
                 model_image=model_image,
                 model_time_series1=model_time_series1,
                 model_time_series2=model_time_series2):
        super(MultiModelAttention, self).__init__()
        self.model_image = model_image
        self.model_time_series1 = model_time_series1
        self.model_time_series2 = model_time_series2

        self.feature_size = features_size
        self.num_model = 3
        self.bn1 = nn.BatchNorm1d(self.feature_size)
        self.bn2 = nn.BatchNorm1d(self.feature_size)
        self.bn3 = nn.BatchNorm1d(self.feature_size)
        self.attention_weights = nn.Parameter(
            torch.cat((torch.ones(1, self.feature_size), torch.zeros(2, self.feature_size)), dim=0))
        # ( num_features, feature_size), torch.randn(3, self.feature_size)
        self.fc1 = nn.Linear(self.feature_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.dropout = nn.Dropout(0.2)

    def forward(self, img_input, ts_input1, ts_input2):
        _, feature_img = self.model_image(img_input)
        _, feature_ts1 = self.model_time_series1(ts_input1)
        _, feature_ts2 = self.model_time_series2(ts_input2)
        feature_img = self.bn1(feature_img)
        feature_ts1 = self.bn2(feature_ts1)
        feature_ts2 = self.bn3(feature_ts2)

        features = torch.stack([feature_img, feature_ts1, feature_ts2],
                               dim=1)  # (batch_size, num_features, feature_size)

        normalized_weights = F.softmax(self.attention_weights, dim=0)  # softmax attention (num_features, feature_size)

        weighted_features = features * normalized_weights.unsqueeze(0)  # (batch_size, num_features, feature_size)

        fused_features = torch.sum(weighted_features, dim=1)  # (batch_size, feature_size)

        x = self.dropout(torch.relu(self.fc1(fused_features)))  # fc layer with Tanh or relu ????
        x = torch.relu(self.fc2(x))

        return self.fc3(x)

########################################################################################
class MultiModel(nn.Module):
    def __init__(self, features_size,
                 model_image=model_image,
                 model_time_series1=model_time_series1,
                 model_time_series2=model_time_series2):
        super(MultiModel, self).__init__()

        self.feature_size = features_size
        self.model_image = model_image
        self.model_time_series1 = model_time_series1
        self.model_time_series2 = model_time_series2

        self.bn1 = nn.BatchNorm1d(self.feature_size)
        self.bn2 = nn.BatchNorm1d(self.feature_size)
        self.bn3 = nn.BatchNorm1d(self.feature_size)

        self.fc1 = nn.Linear(self.feature_size * 3, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, img_input, ts_input1, ts_input2):
        _, feature_img = self.model_image(img_input)
        _, feature_ts1 = self.model_time_series1(ts_input1)
        _, feature_ts2 = self.model_time_series2(ts_input2)
        feature_img = self.bn1(feature_img)
        feature_ts1 = self.bn2(feature_ts1)
        feature_ts2 = self.bn3(feature_ts2)

        combined_feature = torch.cat((feature_img, feature_ts1, feature_ts2), dim=1)

        output = F.relu(self.fc1(combined_feature))
        output = self.fc2(output)
        return output

########################################################################################
# Instantiate the multi-modal model using the pretrained models
print('################ multi-modal ##################\n')
# multi_model = MultiModelAttention(features_size=feature_size,
#                                  model_image=model_image,
#                                  model_time_series1=model_time_series1,
#                                  model_time_series2=model_time_series2).to(device)

multi_model = MultiModel(features_size=feature_size,
                          model_image=model_image,
                          model_time_series1=model_time_series1,
                          model_time_series2=model_time_series2).to(device)

# Freeze the pretrained models
for param in model_image.parameters():
    param.requires_grad = True
for param in model_time_series1.parameters():
    param.requires_grad = True
for param in model_time_series2.parameters():
    param.requires_grad = True


optimizer_multi = optim.Adam(filter(lambda p: p.requires_grad, multi_model.parameters()),
                             lr=1e-4,
                             weight_decay=1e-5)

criterion_multi = nn.CrossEntropyLoss()
train_multi_loss = []
test_multi_loss = []
for epoch in range(10):
    multi_model.train()
    for images, phase, rss, labels in train_loader:
        images, phase, rss, labels = images.to(device), phase.to(device), rss.to(device), labels.to(device)
        optimizer_multi.zero_grad()
        outputs = multi_model(images, phase, rss)  # Forward pass
        loss = criterion_multi(outputs, labels)
        loss.backward()
        optimizer_multi.step()
    train_multi_loss.append(loss.item())

    multi_model.eval()
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, phase, rss, labels in test_loader:
            images, phase, rss, labels = images.to(device), phase.to(device), rss.to(device), labels.to(device)
            outputs = multi_model(images, phase, rss)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion_multi(outputs, labels)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
        test_multi_loss.append(loss.item())
    print(f'\n Accuracy on the test set: {100 * correct / total}%')

plt.figure()
plt.plot(train_multi_loss, label='Training Loss')
plt.plot(test_multi_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig(os.path.join(os.path.dirname(script_path), "train_multi.png"))

cf_matrix = confusion_matrix(y_true, y_pred)
#    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
#                         columns = [i for i in classes])
#    plt.figure(figsize = (12,7))
#    sn.heatmap(df_cm, annot=True)
#    plt.savefig(os.path.join(os.path.dirname(script_path), 'output.png'))
print(cf_matrix)
