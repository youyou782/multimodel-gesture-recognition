import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
# from Phase_model import LSTMModel
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")


class LSTMModel_phase(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=10, num_layers=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions


class LSTMModel_rss(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=10, num_layers= 2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions
    
class MultiModel(nn.Module):
    def __init__(self, model_image, model_time_series1, model_time_series2):
        super(MultiModel, self).__init__()

        self.model_image = model_image
        self.model_time_series1 = model_time_series1
        self.model_time_series2 = model_time_series2
        # self.lstm = nn.LSTM()
        
        self.classifier = nn.Linear(28, 4)  

    def forward(self, img_input, ts_input1, ts_input2):

        feature_img = self.model_image(img_input)
        feature_ts1 = self.model_time_series1(ts_input1)
        feature_ts2 = self.model_time_series2(ts_input2)

        combined_feature = torch.cat((feature_img, feature_ts1, feature_ts2), dim=1)
        
        output = self.classifier(combined_feature)
        return output

class MultiModelAttention(nn.Module):
    def __init__(self, model_image, model_time_series1, model_time_series2):
        super(MultiModelAttention, self).__init__()
        self.model_image = model_image
        self.model_time_series1 = model_time_series1
        self.model_time_series2 = model_time_series2

        self.feature_size = 10
        self.num_model = 3
        self.attention_weights = nn.Parameter(torch.randn(3, 10))  # (batch_size, num_features, feature_size)
        self.fc1 = nn.Linear(self.feature_size, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, img_input, ts_input1, ts_input2):
        feature_img = self.model_image(img_input)
        feature_ts1 = self.model_time_series1(ts_input1)
        feature_ts2 = self.model_time_series2(ts_input2)
        features = torch.stack([feature_img, feature_ts1, feature_ts2], dim=1)  # (batch_size, num_features, feature_size)

        normalized_weights = F.softmax(self.attention_weights, dim=0)  # softmax attention (num_features, feature_size)

        weighted_features = features * normalized_weights.unsqueeze(0)  # (batch_size, num_features, feature_size)

        fused_features = torch.sum(weighted_features, dim=1)  # (batch_size, feature_size)

        x = torch.relu(self.fc1(fused_features))  # fc layer with Tanh or relu ????
        output = F.softmax(self.fc2(x), dim=1)  # Softmax to get probabilities

        return output

class MultiModalDataset(Dataset):
    def __init__(self, image, phase, rss, label):
        self.image = image
        self.phase = phase
        self.rss = rss
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = self.image[idx]
        phase = self.phase[idx]
        rss = self.rss[idx]
        label = self.label[idx]
        # 转换为torch.Tensor
        # image_tensor = torch.tensor(image, dtype=torch.float32)
        # phase_tensor = torch.tensor(phase, dtype=torch.float32)
        # rss_tensor = torch.tensor(rss, dtype=torch.float32)
        # label_tensor = torch.tensor(label, dtype=torch.long)
        return image, phase, rss, label

# image model 
vgg16 = models.vgg16(pretrained=False)
vgg16.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
vgg16.classifier[3] = nn.Linear(4096, 10)
classifier_list = list(vgg16.classifier.children())
classifier_list = classifier_list[:-1]

vgg16.classifier = nn.Sequential(*classifier_list)
# num_features = vgg16.classifier[2].in_features
# vgg16.classifier[4] = nn.Linear(num_features, 20)


# vgg16.load_state_dict(torch.load("vgg16_without_last_layer_weights.pth"))


model_image = vgg16.to(device)
# multimodel = MultiModalModel(model_image, model_time_series1, model_time_series2)

#time_series_model 1/phase model

model_time_series1 = LSTMModel_phase().to(device)
# model_time_series1.load_state_dict(torch.load("Phase_LSTM2_parameter.pt"))

#time_series_model 2/rss model

model_time_series2 = LSTMModel_rss().to(device)
# model_time_series2.load_state_dict(torch.load("RSS_LSTM2_parameter.pt"))

if __name__ == "__main__":
    
    # model = MultiModel(model_image, model_time_series1, model_time_series2).to(device)
    model = MultiModelAttention(model_image, model_time_series1, model_time_series2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    classes = ["LIFT", "LR", "PUSH", "PULL"]
    script_path = os.path.abspath(__file__)
    
    # image_path = os.path.join(os.path.dirname(script_path), "../DataSet/Composed_Image")
    # phase_path = os.path.join(os.path.dirname(script_path), "../DataSet/Phase_v4")
    # rss_path = os.path.join(os.path.dirname(script_path), "../DataSet/Rss_v4")
    
    image_path = os.path.join(os.path.dirname(script_path), "image_data.npy")
    phase_path = os.path.join(os.path.dirname(script_path), "v4_Phase_data.npy")
    rss_path = os.path.join(os.path.dirname(script_path), "v4_RSS_data.npy")
    label_path = os.path.join(os.path.dirname(script_path), "image_label.npy")
    
     
    image = np.load(image_path)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(1)
    print("image.size: ", image.size())
    
    phase = np.load(phase_path)
    phase = torch.tensor(phase, dtype=torch.float32).unsqueeze(-1)
    print("phase.size: ", phase.size())
    
    rss = np.load(rss_path)
    rss = torch.tensor(rss, dtype=torch.float32).unsqueeze(-1)
    print("rss.size: ", rss.size())
    
    label = np.load(label_path)
    
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)
    # print(image.shape, image.size)
    # print(phase.shape, phase.size)
    
    label =torch.tensor(label, dtype=torch.long)
    print("label.size: ", label.size())
    
    
    
    dataset = MultiModalDataset(image, phase, rss, label)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    
    num_epochs = 50
    train_acc_array = []
    test_acc_array = []
    for epoch in range(num_epochs):
        
        model.train()
        train_loop = tqdm(train_loader)
        
        for image, phase, rss, label in train_loader:
            
            image, phase, rss, label = image.to(device), phase.to(device), rss.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(image, phase, rss)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loop.update()
            train_loop.set_description(f'Epoch {epoch+1}/{num_epochs}')
            
        if(epoch % 5 == 0):
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for image, phase, rss, label in train_loader:
                    image, phase, rss, label = image.to(device), phase.to(device), rss.to(device), label.to(device)
                    outputs = model(image, phase, rss)
                    _, predicted = torch.max(outputs.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    
            print(f'\n Accuracy on the train set: {100 * correct / total}%')
            train_acc_array.append(100 * correct / total)
            
            # model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for image, phase, rss, label in test_loader:
                    image, phase, rss, label = image.to(device), phase.to(device), rss.to(device), label.to(device)
                    outputs = model(image, phase, rss)
                    _, predicted = torch.max(outputs.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()

            print(f'\n Accuracy on the test set: {100 * correct / total}%')
            test_acc_array.append(100 * correct / total)
            
    print('Finished Training')
        #process input tensor