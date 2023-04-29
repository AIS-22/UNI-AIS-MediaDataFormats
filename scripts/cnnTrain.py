import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.transforms import Lambda
from torch.autograd import Variable
import os
import torchinfo
import cnnModel
import cnnDataset

def train(model, trainloader, optimizer, criterion, dev):
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
        print(f'Batch: {batch_idx} of {len(trainloader)} \r', end='')
    print('Train Loss: \t {:.6f}'.format( train_loss / len(trainloader.dataset)))

def test(model, testloader, criterion, dev):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (orig, label) in enumerate(testloader):
            orig = orig.to(dev)
            # label to one hot
            label = torch.nn.functional.one_hot(label, num_classes=len(testloader.dataset.classes)).to(torch.float32).to(dev)
            output = model(orig)
            test_loss += criterion(output, label).item()
    test_loss /= len(testloader.dataset)
    print('Test Loss: \t {:.6f}'.format(test_loss))


## create a training and test dataset
#def create_dataset():
#    # create a training dataset from multiple image folders 'Images/DIV2K_train_HR'
#    trainset = torchvision.datasets.ImageFolder(root='Images/DIV2K_train_HR/Decoded/', transform=transforms.ToTensor())
#    # target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
#    # create a test dataset from multiple image folders 'Images/DIV2K_valid_HR'
#    testset = torchvision.datasets.ImageFolder(root='Images/DIV2K_valid_HR/Decoded/', transform=transforms.ToTensor())
#    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
#    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)
#    return trainloader, testloader


if __name__ == "__main__":
    dev = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    trainloader, testloader = cnnDataset.create_dataset()
    # print the content of the dataset
    #file = trainloader.dataset[0][0].numpy().transpose(1, 2, 0)
    #plt.imshow(file)
    #plt.show()
    modelPrint = cnnModel.selfDefinedCNN(num_output=len(trainloader.dataset.classes))
    torchinfo.summary(modelPrint, input_size=(1, 3, 512, 512))

    model = cnnModel.selfDefinedCNN(num_output=len(trainloader.dataset.classes)).to(dev)
    # define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    #model_compile = torch.compile(model)
    epochs = 5

    print("Start training:\n")
    for epoch in range(1, epochs + 1):
        print(f"-------------------------------\nEpoch {epoch}\n----------")
        train(model, trainloader, optimizer, criterion, dev)
        test(model, testloader, criterion, dev)
        print("-------------------------------")
    print("Done!")

    save_model = True
    if save_model:
        torch.save(model.state_dict(), "models/cnnParams.pt")
        print("Model saved")