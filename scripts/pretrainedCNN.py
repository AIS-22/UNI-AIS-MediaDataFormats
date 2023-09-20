import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from sklearn.metrics import precision_recall_fscore_support
import cnnDataset
import evaluateMixedModel

#TODO: Log the output in a file

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1_score:.4f}')


transform = transforms.Compose([
    # the model excepts just 224x224 images, maybe we need to crop our images before encoding them
    # I used crop, resize would change the image and therefore can lead to probles for our classification
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_loader, val_loader = cnnDataset.create_dataset(transform=transform)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

"""
Different models can be used, see examples in
https://pyimagesearch.com/2021/07/26/pytorch-image-classification-with-pre-trained-networks/
"""
# resNet
model_name = 'resnet18'
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_new_classes = 10  # number of codecs

# use this if e.g. resNet get used
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_new_classes)  # Replace the final layer with the number of codec classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# take a low learning rate since, the model is already pretrained
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# change the number of epochs here!
num_epochs = 10
losses = np.zeros((num_epochs, 2))
for epoch in range(num_epochs):
    print(f'Start to train epoch: {epoch + 1}')
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset.samples)
    print(f'Train Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}')

    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # Update the total count of processed samples
            test_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    test_loss /= len(val_loader.dataset.samples)
    print(f'Validation accuracy: {accuracy:.4f} loss: {test_loss} in epoch: {epoch + 1}')
    losses[epoch, 0] = epoch_loss
    losses[epoch, 1] = test_loss
# store the results in a file
np.save('results/losses_mixed_model.npy', losses)

evaluate_model(model, val_loader)

save_model = True
if save_model:
    torch.save(model.state_dict(), "models/cnnParams_" + model_name + ".pt")
    print("Model saved")

evaluateMixedModel.main()
