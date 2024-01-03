import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torchvision import transforms, models
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import cnnDataset


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

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            #l2 norm
            outputs = nn.functional.normalize(outputs, p=2, dim=1)

            all_preds.append(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #scatter plot all_preds color coded by all_labels
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_preds = all_preds.reshape(all_preds.shape[0], all_preds.shape[2])
    pca = PCA(n_components=2)
    pca.fit(all_preds)
    all_preds = pca.transform(all_preds)

    #set label to name from dataloader
    all_labels = np.array([test_loader.dataset.classes[label] for label in all_labels])

    return all_preds, all_labels


def main():
    #filesizes = ['5', '10', '17', '25', '32', '40', '50', '60', '75', '100']
    filesizes = ['10']

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

    for filesize in filesizes:
        transform = transforms.Compose([
            # the model excepts just 224x224 images, maybe we need to crop our images before encoding them
            # I used crop, resize would change the image and therefore can lead to probles for our classification
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        _, val_loader = cnnDataset.create_dataset(transform=transform, filesize=filesize)

        model_name = 'resnet18'
        model = models.resnet18()
        print('Evaluate pretrained model ( ' + model_name + ' ) with Filesize = ' + filesize + ' kB')
        result_dictionary[filesize] = evaluate_model(model, val_loader)

        # store the results in a file
        np.save('results/wo_transfer_results.npy', result_dictionary)


if __name__ == '__main__':
    main()
