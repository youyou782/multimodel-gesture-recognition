import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 数据转换 - 用于单通道图像
transform = transforms.Compose([
    
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# 加载数据集
script_path = os.path.abspath(__file__)

image_path = os.path.join(os.path.dirname(os.path.dirname(script_path)), 'Dataset/Image')
full_dataset = datasets.ImageFolder(root=image_path, transform=transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


# 加载并修改VGG16模型
vgg16 = models.vgg16(pretrained=True)
# for param in vgg16.features.parameters():
#     param.requires_grad = False  # 冻结特征层
vgg16.classifier[3] = nn.Linear(4096,20); 
num_features = vgg16.classifier[3].out_features
vgg16.classifier[6] = nn.Linear(num_features, 4)  # 修改最后一层为4类输出

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.0001, momentum=0.9)

vgg16.to(device)
num_epochs = 100

train_acc_array = []
test_acc_array = []
for epoch in range(num_epochs):
    
    vgg16.train()
    train_loop = tqdm(train_loader)
    
    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loop.update()
        train_loop.set_description(f'Epoch {epoch+1}/{num_epochs}')
        
    if(epoch % 5 == 0):
        
        vgg16.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = vgg16(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        print(f'\n Accuracy on the train set: {100 * correct / total}%')
        train_acc_array.append(100 * correct / total)
        
        
        # vgg16.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = vgg16(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'\n Accuracy on the test set: {100 * correct / total}%')
        test_acc_array.append(100 * correct / total)
        
print('Finished Training')

train_acc_array = np.array(train_acc_array)
test_acc_array = np.array(test_acc_array)

np.savetxt('train_acc_array.txt', train_acc_array, delimiter=',')
np.savetxt('test_acc_array.txt', test_acc_array, delimiter=',')


model_state_dict = vgg16.state_dict()

del model_state_dict['classifier.6.weight']
del model_state_dict['classifier.6.bias']

torch.save(model_state_dict, 'vgg16_without_last_layer_weights.pth')