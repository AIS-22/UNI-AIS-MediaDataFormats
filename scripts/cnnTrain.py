import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.models import resnet152, resnet34, resnet18
from torchvision import models
import os
import torchinfo
import cnnModel
import cnnDataset

def train(model, trainloader, optimizer, criterion, dev):
    model.train()
    train_loss = 0
    for batch_idx, (orig, label) in enumerate(trainloader):
        orig = orig.to(dev)
        label = F.one_hot(label, num_classes=len(trainloader.dataset.classes)).to(torch.float32).to(dev)
        optimizer.zero_grad()
        output = model(orig)
        loss = criterion(output, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        print(f'Batch: {batch_idx} of {len(trainloader)} \r', end='')
    print('Train Loss:     {:.6f}'.format( train_loss / len(trainloader.dataset)))

def test(model, testloader, criterion, dev):
    model.eval()
    # also tset the accuracy
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (orig, label) in enumerate(testloader):
            orig = orig.to(dev)
            label = F.one_hot(label, num_classes=len(testloader.dataset.classes)).to(torch.float32).to(dev)
            output = model(orig)
            test_loss += criterion(output, label).item()
            # also test the accuracy
            correct += (torch.max(output.data, 1)[1] == torch.max(label.data, 1)[1]).sum().item()
    print(' Test Loss:     {:.6f}'.format(test_loss / len(testloader.dataset)))
    print(' Test Accuracy: {:.2f}%'.format(100 * correct / len(testloader.dataset)))

if __name__ == "__main__":
    dev = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    #printModel = True
    #trainModel = False
    printModel = False
    trainModel = True
    loadModel = True
    trainloader, testloader = cnnDataset.create_dataset()
    # print the content of the dataset
    #file = trainloader.dataset[0][0].numpy().transpose(1, 2, 0)
    #plt.imshow(file)
    #plt.show()
    if printModel:
        #modelPrint = cnnModel.CNN(num_output=len(trainloader.dataset.classes))
        modelPrint = resnet152(num_classes=len(trainloader.dataset.classes))
        torchinfo.summary(modelPrint, input_size=(1, 3, 512, 512))
    if trainModel:
        #model = cnnModel.CNN(num_output=len(trainloader.dataset.classes)).to(dev)
        #model = cnnModel.BasicNet().to(dev)
        if loadModel:
            model = resnet18(num_classes=len(trainloader.dataset.classes)).to(dev)
            model.load_state_dict(torch.load("models/cnnParams.pt"))
        else:
            model = resnet18(num_classes=len(trainloader.dataset.classes)).to(dev)
        # define the loss function and the optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        #model_compile = torch.compile(model)
        epochs = 20

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