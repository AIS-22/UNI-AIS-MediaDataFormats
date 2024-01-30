import torch
import torchvision
from torchvision import transforms
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# create a training and test dataset


def create_dataset(transform=transforms.ToTensor(), filesize="all"):
    # create a training dataset from multiple image folders 'Images/DIV2K_train_HR'
    trainset = torchvision.datasets.ImageFolder(root='Images/DIV2K_train_HR/Decoded/'+filesize+'/', transform=transform)
    # target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    # create a test dataset from multiple image folders 'Images/DIV2K_valid_HR'
    testset = torchvision.datasets.ImageFolder(root='Images/DIV2K_valid_HR/Decoded/'+filesize+'/', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    return trainloader, testloader
