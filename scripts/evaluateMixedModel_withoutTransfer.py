import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torchvision import transforms, models
import numpy as np
from matplotlib import pyplot as plt

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
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            #L2 normalization of the output
            prediction = np.sqrt(np.sum(np.square(outputs.cpu().numpy()), axis=1))
            prediction = np.sqrt(np.sum(np.square(prediction), axis=0))
            total += labels.size(0)  # Update the total count of processed samples

            all_preds.append(prediction)
            all_labels.extend(labels.cpu().numpy())

    #scatter plot all_preds color coded by all_labels
    
    plt.scatter(all_preds,range(len(all_preds)))
    plt.show()
    plt.savefig('results/scatter_plot_withoutTrain.png')


    return 


def main():
    #filesizes = ['5', '10', '17', '25', '32', '40', '50', '60', '75', '100']
    filesizes = ['5']

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
        #num_ftrs = model.fc.in_features
        #num_new_classes = 6
        #model.fc = nn.Linear(num_ftrs, num_new_classes)  # Replace the final layer with the number of codec classes
        #model.load_state_dict(torch.load('models/cnnParams_resnet18.pt'))
        print('Evaluate pretrained model ( ' + model_name + ' ) with Filesize = ' + filesize + ' kB')
        result_dictionary[filesize] = evaluate_model(model, val_loader)

        # store the results in a file
        np.save('results/mixed_results.npy', result_dictionary)


if __name__ == '__main__':
    main()
