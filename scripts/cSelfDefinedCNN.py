import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.transforms import Lambda
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torchinfo

class selfDefinedCNN(nn.Module):
    def __init__(self, num_output=4):
        super(selfDefinedCNN, self).__init__()
        # layer from 512x512x3 to 512x512x64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        # layer from 512x512x64 to 256x256x128
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        # layer from 256x256x128 to 256x256x128
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        # layer from 256x256x128 to 128x128x256
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        # layer from 128x128x256 to 128x128x256
        #self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        # layer from 128x128x256 to 64x64x512
        self.conv6 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        # layer from 64x64x512 to 64x64x512
        #self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        # layer from 64x64x512 to 32x32x512
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        # layer from 32x32x512 to 32x32x512
        #self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        # layer from 32x32x512 to 16x16x512
        self.conv10 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        # layer from 16x16x512 to 16x16x512
        #self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        # layer from 16x16x512 to 8x8x512
        self.conv12 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        # layer from 8x8x512 to 8x8x512
        #self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        # layer from 8x8x512 to 4x4x512
        self.conv14 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        # layer from 4x4x512 to 1x1x512
        self.conv15 = nn.Conv2d(512, 512, 4)
        # fully connected layer
        self.fc1 = nn.Linear(512, num_output)
        # 01 512x512x3 -> 512x512x64
        # 02 512x512x64 -> 256x256x128
        # 03 256x256x128 -> 256x256x128
        # 04 256x256x128 -> 128x128x256
        # 05 128x128x256 -> 128x128x256
        # 06 128x128x256 -> 64x64x512
        # 07 64x64x512 -> 64x64x512
        # 08 64x64x512 -> 32x32x512
        # 09 32x32x512 -> 32x32x512
        # 10 32x32x512 -> 16x16x512
        # 11 16x16x512 -> 16x16x512
        # 12 16x16x512 -> 8x8x512
        # 13 8x8x512 -> 8x8x512
        # 14 8x8x512 -> 4x4x512
        # 15 4x4x512 -> 1x1x512

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 512x512x3 -> 512x512x64
        x = F.relu(self.conv2(x)) # 512x512x64 -> 256x256x128
        #x = F.relu(self.conv3(x)) # 256x256x128 -> 256x256x128
        x = F.relu(self.conv4(x)) # 256x256x128 -> 128x128x256
        #x = F.relu(self.conv5(x)) # 128x128x256 -> 128x128x256
        x = F.relu(self.conv6(x)) # 128x128x256 -> 64x64x512
        #x = F.relu(self.conv7(x)) # 64x64x512 -> 64x64x512
        x = F.relu(self.conv8(x)) # 64x64x512 -> 32x32x512
        #x = F.relu(self.conv9(x)) # 32x32x512 -> 32x32x512
        x = F.relu(self.conv10(x)) # 32x32x512 -> 16x16x512
        #x = F.relu(self.conv11(x)) # 16x16x512 -> 16x16x512
        x = F.relu(self.conv12(x)) # 16x16x512 -> 8x8x512
        #x = F.relu(self.conv13(x)) # 8x8x512 -> 8x8x512
        x = F.relu(self.conv14(x)) # 8x8x512 -> 4x4x512
        x = F.relu(self.conv15(x)) # 4x4x512 -> 1x1x512
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        return x

def train(epoch, model, trainloader, optimizer, criterion, dev):
    model.train()
    train_loss = 0
    for batch_idx, (orig, label) in enumerate(trainloader):
        orig = orig.to(dev)
        #label = label.to(dev)
        # label to one hot
        label = torch.nn.functional.one_hot(label, num_classes=len(trainloader.dataset.classes)).to(torch.float32).to(dev)
        optimizer.zero_grad()
        output = model(orig)
        loss = criterion(output, label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('Train Epoch:{} \t Loss: {:.6f}'.format(epoch, train_loss / len(trainloader.dataset)))

def test(epoch, model, testloader, criterion, dev):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in testloader:
            data = data.to(dev)
            output = model(data)
            test_loss += criterion(output, data).item()
    test_loss /= len(testloader.dataset)
    print('\t Test \t Loss: {:.6f}'.format(test_loss))


# create a training and test dataset
def create_dataset():
    # list fodler names
    folders = os.listdir('Images/DIV2K_train_HR/')
    # remove all files that are in the folder
    folders = [f for f in folders if os.path.isdir(os.path.join('Images/DIV2K_train_HR/', f))]
    print('Train folders: ', folders)

    # create a training dataset from multiple image folders 'Images/DIV2K_train_HR'
    trainset = torchvision.datasets.ImageFolder(root='Images/DIV2K_train_HR/Decoded/', transform=transforms.ToTensor())
    # target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    # create a test dataset from multiple image folders 'Images/DIV2K_valid_HR'
    testset = torchvision.datasets.ImageFolder(root='Images/DIV2K_valid_HR/Decoded/', transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)
    return trainloader, testloader


if __name__ == "__main__":
    dev = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    trainloader, testloader = create_dataset()
    # print the content of the dataset
    #file = trainloader.dataset[0][0].numpy().transpose(1, 2, 0)
    #plt.imshow(file)
    #plt.show()
    modelPrint = selfDefinedCNN(num_output=len(trainloader.dataset.classes))
    print(torchinfo.summary(modelPrint, input_size=(1, 3, 512, 512)))

    model = selfDefinedCNN(num_output=len(trainloader.dataset.classes)).to(dev)
    # define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    #print(f"Model structure: {model}\n\n")

    #for name, param in model.named_parameters():
    #    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    for epoch in range(1, epochs + 1):
        train(epoch, model, trainloader, optimizer, criterion, dev)
        test(epoch, model, testloader, criterion, dev)
