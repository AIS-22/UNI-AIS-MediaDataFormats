import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from sklearn.metrics import precision_recall_fscore_support
import cnnDataset


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

filesizes = ['40', '50', '60', '75', '100']

for filesize in filesizes:
    transform = transforms.Compose([
        # the model excepts just 224x224 images, maybe we need to crop our images before encoding them
        # I used crop, resize would change the image and therefore can lead to probles for our classification
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader, val_loader = cnnDataset.create_dataset(transform=transform, filesize=filesize)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    modelName ='cnnParams_vgg16.pt'
    model = torch.load('models/' + modelName)
    print('Evaluate pretrained model ( ' + modelName + ' ) with Filesize = ' + filesize + ' kB')
    evaluate_model(model, val_loader)