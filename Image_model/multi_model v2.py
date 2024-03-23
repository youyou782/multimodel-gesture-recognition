import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from Phase_model import LSTMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

def matrix():
    model.eval()
    labels = []
    preds = []
    with torch.no_grad():
        for image, phase, rss, label in test_loader:
            image, phase, rss, label = image.to(device), phase.to(device), rss.to(device), label.to(device)
            outputs = model(image, phase, rss)
            _, predicted = torch.max(outputs.data, 1)
            
            # 收集预测和真实标签
            labels.extend(label.cpu().numpy())
            preds.extend(predicted.cpu().numpy())

        # 计算混淆矩阵
        cm = confusion_matrix(labels, preds)
        print(cm)

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm_normalized)
        # 绘制混淆矩阵
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', ax=ax, cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)

        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig(f"./v2/multimodel_v2_matrix_e{epoch}.png")
                
    print('Finished drawing')
    
class LSTMModel_phase(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=20,num_layers=2):
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
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=20,num_layers = 2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers , batch_first=True)
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
        
        self.classifier = nn.Linear(60, 4)  

    def forward(self, img_input, ts_input1, ts_input2):

        feature_img = self.model_image(img_input)
        feature_ts1 = self.model_time_series1(ts_input1)
        feature_ts2 = self.model_time_series2(ts_input2)

        combined_feature = torch.cat((feature_img, feature_ts1, feature_ts2), dim=1)
        
        output = self.classifier(combined_feature)
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
vgg16.classifier[3] = nn.Linear(4096,20); 
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
    
    model = MultiModel(model_image, model_time_series1, model_time_series2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.4)
    
    classes = ["LIFT", "LR", "PUSH", "PULL"]
    script_path = os.path.abspath(__file__)
    
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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    num_epochs = 300
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
            scheduler.step(correct / total)
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
            if correct / total > 0.85 :
                torch.save(model, f'./v2/multimode_v2_para_e{epoch}.pth')
                matrix()
                # break
    
    matrix()
    
    #last test to draw confusion 
