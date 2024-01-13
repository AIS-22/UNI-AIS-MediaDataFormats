import matplotlib.pyplot as plt
import matplotlib
#Make matplotlib use latex for font rendering
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib

RESULTS_FOLDER = 'results/'
AVG_FILESIZE = 470

#CONST for plotting so all plots are the same
def set_figsize():
    plt.figure(figsize=(7, 5))
    plt.rcParams['font.size'] = 12


def plot_accuracy_results():
    filesizes = ['5', '10', '17', '25', '32', '40', '50', '60', '75', '100']

    # load dic from file
    mixed_results = np.load(RESULTS_FOLDER + 'accuracy_mixed_model.npy', allow_pickle=True).item()
    keys = [int(key) for key in mixed_results.keys()]

    # Map file sizes to colors
    color_dict = {
        'pre': plt.get_cmap('Paired')(1),
        'self': plt.get_cmap('Paired')(3),
        '5':  plt.get_cmap('Paired')(5),
        '10': plt.get_cmap('Paired')(7),
        '17': plt.get_cmap('Paired')(9),
        '25': plt.get_cmap('Dark2')(6),
        '32': plt.get_cmap('Accent')(5),
        '40': plt.get_cmap('Accent')(7),
        '50': plt.get_cmap('Set2')(4),
        '60': plt.get_cmap('tab10')(9),
        '75': plt.get_cmap('tab20b')(19),
        '100': plt.get_cmap('Pastel1')(0)
    }

    cmap = plt.get_cmap('tab20')
    set_figsize()
    plt.plot(keys, mixed_results.values(), color=cmap(0), label='Mixed pre trained (5-32)')
     
    mixed_self_results = np.load(RESULTS_FOLDER + 'accuracy_mixed_self_model.npy', allow_pickle=True).item()
    keys = [int(key) for key in mixed_self_results.keys()]
    plt.plot(keys, mixed_self_results.values(), color=color_dict['self'], label='Mixed self trained (5-32)')

    for filesize in filesizes:
        res_dict = np.load(RESULTS_FOLDER + 'accuracy_fs_' + filesize + '_model.npy', allow_pickle=True).item()
        keys = [int(key) for key in res_dict.keys()]
        plt.plot(keys, res_dict.values(), color=color_dict[filesize],
                 label=filesize + f'  c-rate: {(AVG_FILESIZE / float(filesize)):.2f}')

    plt.grid()
    plt.xlabel('Test Filesize')
    plt.ylabel('Accuracy')
    plt.legend(title='Trained file sizes (mean c-rate)')
    plt.savefig('Plots/accuracy/accuracy_comparison.pgf')
    plt.close()

    filesizes = ['10', '25', '40', '100']

    # load dic from file
    mixed_results = np.load(RESULTS_FOLDER + 'accuracy_mixed_model.npy', allow_pickle=True).item()
    keys = [int(key) for key in mixed_results.keys()]
    cmap = plt.get_cmap('tab20')
    set_figsize()
    plt.plot(keys, mixed_results.values(), color=color_dict['pre'], label='Mixed pre trained (5-32)')

    mixed_self_results = np.load(RESULTS_FOLDER + 'accuracy_mixed_self_model.npy', allow_pickle=True).item()
    keys = [int(key) for key in mixed_self_results.keys()]
    plt.plot(keys, mixed_self_results.values(), color=color_dict['self'], label='Mixed self trained (5-32)')

    for filesize in filesizes:
        res_dict = np.load(RESULTS_FOLDER + 'accuracy_fs_' + filesize + '_model.npy', allow_pickle=True).item()
        keys = [int(key) for key in res_dict.keys()]
        plt.plot(keys, res_dict.values(), color=color_dict[filesize],
                 label=filesize + f'  c-rate: {(AVG_FILESIZE / float(filesize)):.2f}')

    plt.grid()
    plt.xlabel('Test Filesize')
    plt.ylabel('Accuracy')
    plt.legend(title='Trained file sizes (mean c-rate)')
    plt.savefig('Plots/accuracy/accuracy_comparison_filtered.pgf')
    plt.close()


def plot_confusion_matrix():
    filesizes = ['5', '10', '17', '25', '32', '40', '50', '60', '75', '100']
    categories = ['AVIF', 'BPG', 'HEIC', 'JPEG', 'JPEG2000', 'JPEG_XL', 'JPEG_XR_0', 'JPEG_XR_1', 'JPEG_XR_2', 'WEBP']

    # mixed model
    # load dic from file
    mixed_results = np.load(RESULTS_FOLDER + 'conf_matrix_mixed_model.npy', allow_pickle=True).tolist()
    for fs in filesizes:
        set_figsize()
        #sn.set(font_scale=1.4)
        sn.heatmap(mixed_results[fs], vmin=0, vmax=100,
                   xticklabels=categories,
                   yticklabels=categories,
                   annot=True,
                   cmap='Blues',
                   fmt=".0f")
        plt.xticks(rotation=45)
        plt.savefig('Plots/conf_matrix/mixed/conf_matrix_mixed_model_fs_' + fs + '.pgf')
        plt.close()


    # mixed self model
    # load dic from file
    mixed_results = np.load(RESULTS_FOLDER + 'conf_matrix_mixed_self_model.npy', allow_pickle=True).item()

    for fs in filesizes:
        set_figsize()
        #sn.set(font_scale=1.4)
        sn.heatmap(mixed_results[fs], vmin=0, vmax=100,
                   xticklabels=categories,
                   yticklabels=categories,
                   annot=True,
                   cmap='Blues',
                   fmt=".0f")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('Plots/conf_matrix/mixed_self/conf_matrix_mixed_self_model_fs_' + fs + '.pgf')
        plt.close()


    # filesize model
    for fs in filesizes:
        # load dic from file
        mixed_results = np.load(RESULTS_FOLDER + 'conf_matrix_fs_' + fs + '_model.npy', allow_pickle=True).item()
        for ev_size in filesizes:
            keys = [int(key) for key in mixed_results.keys()]
            cmap = plt.get_cmap('tab20')
            set_figsize()
            #sn.set(font_scale=1.4)
            sn.heatmap(mixed_results[ev_size], vmin=0, vmax=500,
                       xticklabels=categories,
                       yticklabels=categories,
                       annot=True,
                       cmap='Blues',
                       fmt=".0f",)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('Plots/conf_matrix/' + fs + '/conf_matrix_fs_'+ ev_size + '.pgf')
            plt.close()

def plot_confusion_matrix_all():
    filesizes = ['5', '10', '17', '25', '32', '40', '50', '60', '75', '100']
    categories = ['AVIF', 'BPG', 'HEIC', 'JPEG', 'JPEG2000', 'JPEG_XL', 'JPEG_XR_0', 'JPEG_XR_1', 'JPEG_XR_2', 'WEBP']

    # mixed model
    # load dic from file
    mixed_results = np.load(RESULTS_FOLDER + 'conf_matrix_all_mixed_model.npy', allow_pickle=True).tolist()
    cmap = plt.get_cmap('tab20')
    
    set_figsize()
    #sn.set(font_scale=1.4)
    sn.heatmap(mixed_results, vmin=0, vmax=1000,
               xticklabels=categories,
               yticklabels=categories,
               annot=True,
               cmap='Blues',
               fmt=".0f")
    plt.xticks(rotation=45)
    plt.savefig('Plots/conf_matrix/mixed/conf_matrix_all_mixed_model.pgf')
    plt.close()

    # mixed self model
    # load dic from file
    mixed_self_results = np.load(RESULTS_FOLDER + 'conf_matrix_all_mixed_self_model.npy', allow_pickle=True).tolist()
    
    set_figsize()
    #sn.set(font_scale=1.4)
    sn.heatmap(mixed_self_results, vmin=0, vmax=1000,
               xticklabels=categories,
               yticklabels=categories,
               annot=True,
               cmap='Blues',
               fmt=".0f")
    plt.xticks(rotation=45)
    plt.savefig('Plots/conf_matrix/mixed_self/conf_matrix_all_mixed_self_model.pgf')
    plt.close()

    # filesize models    
    for fs in filesizes:
        # load dic from file
        fs_results = np.load(RESULTS_FOLDER + 'conf_matrix_all_fs_' + fs + '_model.npy', allow_pickle=True).tolist()
        set_figsize()
        sn.set(font_scale=1.4)
        sn.heatmap(fs_results, vmin=0, vmax=5000,
                   xticklabels=categories,
                   yticklabels=categories,
                   annot=True,
                   cmap='Blues',
                   fmt=".0f")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('Plots/conf_matrix/' + fs + '/conf_matrix_all_fs_' + fs + '.pgf')
        plt.close()


def plot_loss_results():
    filesizes = ['5', '10', '17', '25', '32', '40', '50', '60', '75', '100']

    # mixed model
    # load dic from file
    mixed_loss = np.load(RESULTS_FOLDER + 'losses_mixed_model.npy', allow_pickle=True)
    set_figsize()
    plt.plot(range(1, 11), mixed_loss[:, 0], c='r', label='Train Loss')
    plt.plot(range(1, 11), mixed_loss[:, 1], c='g', label='Validation Loss')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(title='Loss Functions')
    plt.savefig('Plots/loss_comparison/loss_comparison_mixed_model.pgf')
    plt.close()

    mixed_self_loss = np.load(RESULTS_FOLDER + 'losses_mixed_self_model.npy', allow_pickle=True)
    set_figsize()
    plt.plot(range(1, 11), mixed_self_loss[:, 0], c='r', label='Train Loss')
    plt.plot(range(1, 11), mixed_self_loss[:, 1], c='g', label='Validation Loss')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(title='Loss Functions')
    plt.savefig('Plots/loss_comparison/loss_comparison_mixed_self_model.pgf')
    plt.close()

    # filesize models
    for i, filesize in enumerate(filesizes):
        # load dic from file
        fs_model_loss = np.load(RESULTS_FOLDER + 'losses_'+filesize+'_model.npy', allow_pickle=True)
        set_figsize()
        plt.plot(range(1, 11), fs_model_loss [:, 0], c='r', label='Train Loss')
        plt.plot(range(1, 11), fs_model_loss [:, 1], c='g', label='Validation Loss')
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(title='Loss Functions')
        plt.savefig('Plots/loss_comparison/loss_comparison_fs_' + filesize + '_model.pgf')
        plt.close()

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

    set_figsize()
    plt.bar(exec_time_dict.keys(), exec_time_dict.values())
    plt.title('Encoding-Decoding Time Comparison of Huts Image')
    plt.grid()
    plt.xlabel('Algorithm')
    plt.ylabel('Time (s)')
    plt.savefig('Plots/encoding_time_comparison.pgf')
    plt.close()

def plot_filesize_to_target():
    # read csv
    df = pd.read_csv('filesize_log.txt', sep=',', header=None)
    df.columns = ['filesize', 'target', 'codec']
    # sort by string in codec column and reset index
    df = df.sort_values(by=['codec']).reset_index(drop=True)
    # remove / from codec column
    df['codec'] = df['codec'].str.replace('/', '')
    #scatter plot with each codec in different color with filesize as y and index as x
    set_figsize()
    #plt df with a color for each codec
    label = df['codec'].unique()
    # box plot
    plot = plt.boxplot([df[df['codec'] == label]['filesize'] for label in label], labels=label)

    # for i, label in enumerate(label):
    #    plt.scatter(df[df['codec'] == label].index, df[df['codec'] == label]['filesize'], label=label)
    # plot line at target value
    plt.axhline(y=32, color='green', linestyle='--', label='Target')
    # legend
    plt.legend(title='Codec')
    plt.grid()
    plt.xlabel('Codec')
    plt.ylabel('Filesize (KB)')
    plt.xticks(rotation=45)
    plt.savefig('Plots/filesize_to_target.pgf')
    plt.close()

def plot_scatter_without_transfer():
    #read data from npy file
    results = np.load('results/wo_transfer_results.npy', allow_pickle=True).item()

    for filesize in results.keys():
        try:
            all_preds = np.array(results[filesize][0])
            all_labels = np.array(results[filesize][1])
            
            #get all unique labels
            unique_labels = np.unique(all_labels)
            
            set_figsize()
            
            for i in range(len(unique_labels)):
                plt.scatter(all_preds[all_labels == unique_labels[i], 0], all_preds[all_labels == unique_labels[i], 1], label=unique_labels[i])
            plt.legend()
            plt.savefig(f"Plots/plot_scatter_without_transfer_{filesize}.pgf")
            plt.close()

        except:
            continue
        

def main():

    matplotlib.use("pgf")
    matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    })

    #plot_accuracy_results()
    #plot_loss_results() 
    plot_filesize_to_target()
    #plot_confusion_matrix()
    #plot_confusion_matrix_all()
    #plot_dec_enc_time()
    #plot_scatter_without_transfer()


if __name__ == '__main__':
    main()
