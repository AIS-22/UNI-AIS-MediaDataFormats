import torch
import torch.nn as nn
import torch.nn.functional as F

class selfDefinedCNN(nn.Module):
    def __init__(self, num_output=4):
        super(selfDefinedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1) # 512x512x3 -> 512x512x64
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # 512x512x64 -> 256x256x128
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1) # 256x256x128 -> 256x256x128
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # 256x256x128 -> 128x128x256
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1) # 128x128x256 -> 128x128x256
        self.conv6 = nn.Conv2d(256, 512, 3, stride=2, padding=1) # 128x128x256 -> 64x64x512
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1) # 64x64x512 -> 64x64x512
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1) # 64x64x512 -> 32x32x512
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1) # 32x32x512 -> 32x32x512
        self.conv10 = nn.Conv2d(512, 512, 3, stride=2, padding=1) # 32x32x512 -> 16x16x512
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1) # 16x16x512 -> 16x16x512
        self.conv12 = nn.Conv2d(512, 512, 3, stride=2, padding=1) # 16x16x512 -> 8x8x512
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1) # 8x8x512 -> 8x8x512
        self.conv14 = nn.Conv2d(512, 512, 3, stride=2, padding=1) # 8x8x512 -> 4x4x512
        self.conv15 = nn.Conv2d(512, 512, 4) # 4x4x512 -> 1x1x512
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_output)
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
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
    def loadModel(self, model_path):
        self.load_state_dict(torch.load(model_path))
        print('Model loaded from {}'.format(model_path))

class CNN(nn.Module):
    def __init__(self, num_output=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(512, 1024, 3, padding=1, stride=2)
        self.conv8 = nn.Conv2d(1024, 1024, 3, padding=1, stride=2)
        self.conv9 = nn.Conv2d(1024, 1024, 4)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, num_output)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = torch.flatten(x, 1)
      #  x = F.dropout(F.relu(self.fc1(x)), training=self.training)
       # x = F.dropout(F.relu(self.fc2(x)), training=self.training)
        #x = F.dropout(F.relu(self.fc3(x)), training=self.training)
        x = F.softmax(self.fc4(x), dim=1)
        return x
    
    def loadModel(self, model_path):
        self.load_state_dict(torch.load(model_path))
        print('Model loaded from {}'.format(model_path))

class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, 2)
        self.conv5 = nn.Conv2d(64, 128, 3, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 1152)
        self.fc12 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc12(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output