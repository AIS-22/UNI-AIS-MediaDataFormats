import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torchvision import transforms, models
import numpy as np

import cnnDataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def evaluate_model(model, test_loader):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    confusion_matrix = np.zeros((10,10))
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # Update the total count of processed samples
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    accuracy = correct / total
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1_score:.4f}')

    return accuracy, confusion_matrix


def main():
    filesizes = ['5', '10', '17', '25', '32', '40', '50', '60', '75', '100']
    
    model_name = 'cnnParams_self_resnet18.pt'
    model = torch.load('models/' + model_name)

    result_dictionary = {
        '5': 0,
        '10': 0,
        '17': 0,
        '25': 0,
        '32': 0,
        '40': 0,
        '50': 0,
        '60': 0,
        '75': 0,
        '100': 0
    }

    conf_matrix_dictionary = {
        '5': np.zeros((10,10)),
        '10': np.zeros((10,10)),
        '17': np.zeros((10,10)),
        '25': np.zeros((10,10)),
        '32': np.zeros((10,10)),
        '40': np.zeros((10,10)),
        '50': np.zeros((10,10)),
        '60': np.zeros((10,10)),
        '75': np.zeros((10,10)),
        '100': np.zeros((10,10))
    }

    for filesize in filesizes:
        transform = transforms.Compose([
            # the model excepts just 224x224 images, maybe we need to crop our images before encoding them
            # I used crop, resize would change the image and therefore can lead to probles for our classification
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        _, val_loader = cnnDataset.create_dataset(transform=transform, filesize=filesize)

        
        print('Evaluate pretrained model ( ' + model_name + ' ) with Filesize = ' + filesize + ' kB')
        result_dictionary[filesize], conf_matrix_dictionary[filesize] = evaluate_model(model, val_loader)

        # store the results in a file
        np.save('results/mixed_self_results.npy', result_dictionary)
        np.save('results/conf_matrix_mixed_self_model.npy', conf_matrix_dictionary)

if __name__ == '__main__':
    main()