import matplotlib.pyplot as plt
import numpy as np

RESULTS_FOLDER = 'results/'
AVG_FILESIZE = 470


def plot_accuracy_results():
    filesizes = ['5', '10', '17', '25', '32', '40', '50', '60', '75', '100']

    # load dic from file
    mixed_results = np.load(RESULTS_FOLDER + 'mixed_results.npy', allow_pickle=True).item()
    keys = [int(key) for key in mixed_results.keys()]
    cmap = plt.get_cmap('tab20')
    plt.figure(figsize=(20, 10))
    plt.plot(keys, mixed_results.values(), color=cmap(0), label='Mixed pre trained (5-32)')

    for i, filesize in enumerate(filesizes):
        res_dict = np.load(RESULTS_FOLDER + 'results_' + filesize + '_model.npy', allow_pickle=True).item()
        keys = [int(key) for key in res_dict.keys()]
        plt.plot(keys, res_dict.values(), color=cmap(i + 1),
                 label=filesize + f'  c-rate: {(AVG_FILESIZE / float(filesize)):.2f}')

    plt.title('Accuracy Comparison of the Models')
    plt.grid()
    # TODO: add accuracy of self trained model
    plt.axhline(y=0.1, color='red', linestyle='--', label='Mixed self')
    plt.xlabel('Test Filesize')
    plt.ylabel('Accuracy')
    plt.legend(title='Trained file sizes (mean c-rate)')
    plt.savefig('Plots/accuracy_comparison.png')
    plt.show()

#TODO: add confusion matrix plot

def plot_loss_results():
    # load dic from file
    mixed_results = np.load(RESULTS_FOLDER + 'losses_mixed_model.npy', allow_pickle=True)
    plt.figure(figsize=(20, 10))
    plt.plot(range(1, 11), mixed_results[:, 0], c='r', label='Train Loss')
    plt.plot(range(1, 11), mixed_results[:, 1], c='g', label='Validation Loss')
    plt.title('Loss Comparison of Mixed Model')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(title='Loss Functions')
    plt.savefig('Plots/loss_comparison.png')
    plt.show()

    # load dic from file
    mixed_results = np.load(RESULTS_FOLDER + 'losses_10_model.npy', allow_pickle=True)
    plt.figure(figsize=(20, 10))
    plt.plot(range(1, 11), mixed_results[:, 0], c='r', label='Train Loss')
    plt.plot(range(1, 11), mixed_results[:, 1], c='g', label='Validation Loss')
    plt.title('Loss Comparison of Filesize 10 Model (Others simnilar)')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(title='Loss Functions')
    plt.savefig('Plots/loss_comparison_filesize.png')
    plt.show()


def plot_dec_enc_time():
    exec_time_dict = {
        'avif': 0.24,
        'webP': 0.17,
        'bpg': 0.29,
        'heic': 0.3,
        'jxl': 0.9,
        'jxr': 0.2
        # TODO: Measure also jpeg2000 and jpeg of image (avg of train 0770.png and 0565.png
    }

    plt.figure(figsize=(20, 10))
    plt.bar(exec_time_dict.keys(), exec_time_dict.values())
    plt.title('Encoding-Decoding Time Comparison of Huts Image')
    plt.grid()
    plt.xlabel('Algorithm')
    plt.ylabel('Time (s)')
    plt.savefig('Plots/encoding_time_comparison.png')
    plt.show()


def main():
    plot_accuracy_results()
    plot_loss_results()
    plot_dec_enc_time()


if __name__ == '__main__':
    main()
