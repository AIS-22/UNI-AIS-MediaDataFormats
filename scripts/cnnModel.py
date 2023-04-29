import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc1 = nn.Linear(512, 128)
        # fully connected layer
        self.fc2 = nn.Linear(128, num_output)
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
        x = F.softmax(self.fc2(x))
        return x
    
    def loadModel(self, model_path):
        self.load_state_dict(torch.load(model_path))
        print('Model loaded from {}'.format(model_path))

class CNN(nn.Module):
    def __init__(self, num_output=4):
        super(CNN, self).__init__()
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
        self.fc1 = nn.Linear(512, 128)
        # fully connected layer
        self.fc2 = nn.Linear(128, num_output)
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
        x = F.softmax(self.fc2(x))
        return x
    
    def loadModel(self, model_path):
        self.load_state_dict(torch.load(model_path))
        print('Model loaded from {}'.format(model_path))