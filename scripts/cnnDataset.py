import torch
import torchvision
from torchvision import transforms

# create a training and test dataset
def create_dataset():
    # create a training dataset from multiple image folders 'Images/DIV2K_train_HR'
    trainset = torchvision.datasets.ImageFolder(root='Images/DIV2K_train_HR/Decoded/', transform=transforms.ToTensor())
    # target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    # create a test dataset from multiple image folders 'Images/DIV2K_valid_HR'
    testset = torchvision.datasets.ImageFolder(root='Images/DIV2K_valid_HR/Decoded/', transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)
    return trainloader, testloader