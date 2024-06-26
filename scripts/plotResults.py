import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
# Make matplotlib use latex for font rendering
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

RESULTS_FOLDER = 'results/'
AVG_FILESIZE = 470

# CONST for plotting so all plots are the same


def set_figsize():
    plt.figure(figsize=(7, 5))
    plt.rcParams['font.size'] = 16


def plot_accuracy_results():
    filesizes = ['5', '10', '17', '25', '32', '40', '50', '60', '75', '100']

    # load dic from file
    mixed_results = np.load(RESULTS_FOLDER + 'accuracy_mixed_model.npy', allow_pickle=True).item()
    mixed_results_all = np.load(RESULTS_FOLDER + 'accuracy_fs_all_model.npy', allow_pickle=True).item()
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
    line_dict = {
        'pre': '-',
        'self': '--',
        '5': (0, (1, 1)),
        '10': (0, (7, 4, 1, 4)),
        '17': (0, (7, 1, 1, 5)),
        '25': (0, (7, 5, 1, 1)),
        '32': '-',
        '40': (0, (7, 5, 1, 1, 1, 5)),
        '50': (0, (7, 1, 1, 1, 1, 9)),
        '60': (0, (7, 9, 1, 1, 1, 1)),
        '75': (0, (5, 1)),
        '100': (0, (5, 6))
    }

    cmap = plt.get_cmap('tab20')
    set_figsize()
    plt.plot(keys, mixed_results.values(), color=cmap(1), label='Mixed pre trained (5-32)', linestyle='--')
    plt.plot(keys, mixed_results_all.values(), color=cmap(0), label='Mixed pre trained (5-100)', linestyle='--')

    mixed_self_results = np.load(RESULTS_FOLDER + 'accuracy_mixed_self_model.npy', allow_pickle=True).item()
    keys = [int(key) for key in mixed_self_results.keys()]
    plt.plot(keys, mixed_self_results.values(),
             color=color_dict['self'], label='Mixed self trained (5-32)', linestyle='--')

    for filesize in filesizes:
        res_dict = np.load(RESULTS_FOLDER + 'accuracy_fs_' + filesize + '_model.npy', allow_pickle=True).item()
        keys = [int(key) for key in res_dict.keys()]
        plt.plot(keys, res_dict.values(), color=color_dict[filesize],
                 label=filesize + f'  c-rate: {(AVG_FILESIZE / float(filesize)):.2f}')

    plt.grid()
    plt.xlabel('Test Filesize')
    plt.ylabel('Accuracy')
    plt.legend(title='Trained file sizes (mean c-rate)', loc='upper left', bbox_to_anchor=(1, 1))
    plt.gcf().set_size_inches(9, 5)
    # Use tight_layout to ensure all elements fit within the saved area
    plt.tight_layout()
    plt.savefig('Plots/accuracy/accuracy_comparison.pgf')
    plt.close()

    # plot accuracy without mixed models
    set_figsize()
    for filesize in filesizes:
        res_dict = np.load(RESULTS_FOLDER + 'accuracy_fs_' + filesize + '_model.npy', allow_pickle=True).item()
        keys = [int(key) for key in res_dict.keys()]
        plt.plot(keys, res_dict.values(), color=color_dict[filesize], linestyle=line_dict[filesize],
                 label=filesize + f'  c-rate: {(AVG_FILESIZE / float(filesize)):.2f}')

    plt.grid()
    plt.xlabel('Test Filesize')
    plt.ylabel('Accuracy')
    plt.legend(title='Trained file sizes (mean c-rate)', loc='upper left', bbox_to_anchor=(1, 1))
    plt.gcf().set_size_inches(9, 5)
    # Use tight_layout to ensure all elements fit within the saved area
    plt.tight_layout()
    plt.savefig('Plots/accuracy/accuracy_comparison_withoutMixed.pgf')
    plt.close()

    # plot filtered accuracy
    line_dict = {
        '10': (0, (7, 5, 1, 1, 1, 5)),
        '25': (0, (7, 2, 1, 1, 1, 8)),
        '40': (0, (7, 8, 1, 1, 1, 2)),
        '100': (0, (7, 1, 1, 9, 1, 1))
    }
    # mixed pre trained
    keys = [int(key) for key in mixed_results.keys()]
    set_figsize()
    plt.plot(keys, mixed_results_all.values(),
             color=color_dict['pre'], label='Mixed pre trained (5-100)', linestyle='-')
    plt.plot(keys, mixed_results.values(), color=cmap(1),
             label='Mixed pre trained (5-32)', linestyle=(0, (5, 2)))

    # self trained
    mixed_self_results = np.load(RESULTS_FOLDER + 'accuracy_mixed_self_model.npy', allow_pickle=True).item()
    keys = [int(key) for key in mixed_self_results.keys()]
    plt.plot(keys, mixed_self_results.values(),
             color=color_dict['self'], label='Mixed self trained (5-32)', linestyle=(0, (1, 1)))

    filesizes = ['10', '25', '40', '100']
    for filesize in filesizes:
        res_dict = np.load(RESULTS_FOLDER + 'accuracy_fs_' + filesize + '_model.npy', allow_pickle=True).item()
        keys = [int(key) for key in res_dict.keys()]
        plt.plot(keys, res_dict.values(), color=color_dict[filesize], linestyle=line_dict[filesize],
                 label=filesize + f'  c-rate: {(AVG_FILESIZE / float(filesize)):.2f}')

    plt.grid()
    plt.xlabel('Test Filesize')
    plt.ylabel('Accuracy')
    plt.legend(title='Trained file sizes (mean c-rate)', loc='upper left', bbox_to_anchor=(1, 1))
    plt.gcf().set_size_inches(9, 5)
    # Use tight_layout to ensure all elements fit within the saved area
    plt.tight_layout()
    plt.savefig('Plots/accuracy/accuracy_comparison_filtered.pgf')
    plt.close()


def plot_confusion_matrix():
    filesizes = ['5', '10', '17', '25', '32', '40', '50', '60', '75', '100']
    categories = ['AVIF', 'BPG', 'HEIC', 'JPEG', 'JPEG 2000',
                  'JPEG XL', 'JPEG XR_{0}', 'JPEG XR_{1}', 'JPEG XR_{2}', 'WEBP']

    # mixed model
    # load dic from file
    mixed_results = np.load(RESULTS_FOLDER + 'conf_matrix_mixed_model.npy', allow_pickle=True).tolist()
    for fs in filesizes:
        set_figsize()
        plt.gcf().set_size_inches(9, 7)
        sn.heatmap(mixed_results[fs], vmin=0, vmax=100,
                   xticklabels=categories,
                   yticklabels=categories,
                   annot=True,
                   cmap='Blues',
                   fmt=".0f")
        plt.xticks(rotation=-45)
        labels = plt.gca().get_xticklabels()
        plt.setp(labels, ha='left')
        plt.tight_layout()
        plt.savefig('Plots/conf_matrix/mixed_5_to_32/conf_matrix_mixed_model_fs_' + fs + '.pgf')
        plt.close()

    # mixed model 5 to 100 per fs conf matrix
    # load dic from file
    mixed_results_5_to_100 = np.load('results/conf_matrix_fs_all_model.npy', allow_pickle=True).tolist()
    cmap = plt.get_cmap('tab20')

    for fs in filesizes:
        set_figsize()
        plt.gcf().set_size_inches(9, 7)
        sn.heatmap(mixed_results_5_to_100[fs], vmin=0, vmax=100,
                   xticklabels=categories,
                   yticklabels=categories,
                   annot=True,
                   cmap='Blues',
                   fmt=".0f")
        plt.xticks(rotation=-45)
        labels = plt.gca().get_xticklabels()
        plt.setp(labels, ha='left')
        plt.tight_layout()
        plt.savefig(f'Plots/conf_matrix/mixed_5_to_100/conf_matrix_mixed_model_5_to_100_fs_{fs}.pgf')
        plt.close()

    # mixed self model
    # load dic from file
    mixed_results = np.load(RESULTS_FOLDER + 'conf_matrix_mixed_self_model.npy', allow_pickle=True).item()

    for fs in filesizes:
        set_figsize()
        plt.gcf().set_size_inches(9, 7)
        sn.heatmap(mixed_results[fs], vmin=0, vmax=100,
                   xticklabels=categories,
                   yticklabels=categories,
                   annot=True,
                   cmap='Blues',
                   fmt=".0f")
        plt.xticks(rotation=-45)
        labels = plt.gca().get_xticklabels()
        plt.setp(labels, ha='left')
        plt.tight_layout()
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
            plt.gcf().set_size_inches(9, 7)
            sn.heatmap(mixed_results[ev_size], vmin=0, vmax=500,
                       xticklabels=categories,
                       yticklabels=categories,
                       annot=True,
                       cmap='Blues',
                       fmt=".0f",)
            plt.xticks(rotation=-45)
            labels = plt.gca().get_xticklabels()
            plt.setp(labels, ha='left')
            plt.tight_layout()
            plt.savefig('Plots/conf_matrix/' + fs + '/conf_matrix_fs_' + ev_size + '.pgf')
            plt.close()


def plot_confusion_matrix_all():
    filesizes = ['5', '10', '17', '25', '32', '40', '50', '60', '75', '100']
    categories = ['AVIF', 'BPG', 'HEIC', 'JPEG', 'JPEG 2000',
                  'JPEG XL', 'JPEG XR_{0}', 'JPEG XR_{1}', 'JPEG XR_{2}', 'WEBP']

    # mixed model 5 - 32
    # load dic from file
    mixed_results = np.load(RESULTS_FOLDER + 'conf_matrix_all_mixed_model.npy', allow_pickle=True).tolist()
    cmap = plt.get_cmap('tab20')

    set_figsize()
    plt.gcf().set_size_inches(9, 7)
    sn.heatmap(mixed_results, vmin=0, vmax=1000,
               xticklabels=categories,
               yticklabels=categories,
               annot=True,
               cmap='Blues',
               fmt=".0f")
    plt.xticks(rotation=-45)
    labels = plt.gca().get_xticklabels()
    plt.setp(labels, ha='left')
    plt.tight_layout()
    plt.savefig('Plots/conf_matrix/mixed_5_to_32/conf_matrix_all_mixed_model.pgf')
    plt.close()

    # mixed model 5 to 100 all fs conf matrix
    # load dic from file
    mixed_results_5_to_100 = np.load('results/conf_matrix_all_fs_all_model.npy', allow_pickle=True).tolist()
    cmap = plt.get_cmap('tab20')

    set_figsize()
    plt.gcf().set_size_inches(9, 7)
    sn.heatmap(mixed_results_5_to_100, vmin=0, vmax=1000,
               xticklabels=categories,
               yticklabels=categories,
               annot=True,
               cmap='Blues',
               fmt=".0f")
    plt.xticks(rotation=-45)
    labels = plt.gca().get_xticklabels()
    plt.setp(labels, ha='left')
    plt.tight_layout()
    plt.savefig('Plots/conf_matrix/mixed_5_to_100/conf_matrix_mixed_model_5_to_100_fs_all.pgf')
    plt.close()

    # mixed self model
    # load dic from file
    mixed_self_results = np.load(RESULTS_FOLDER + 'conf_matrix_all_mixed_self_model.npy', allow_pickle=True).tolist()

    set_figsize()
    plt.gcf().set_size_inches(9, 7)
    sn.heatmap(mixed_self_results, vmin=0, vmax=1000,
               xticklabels=categories,
               yticklabels=categories,
               annot=True,
               cmap='Blues',
               fmt=".0f")
    plt.xticks(rotation=-45)
    labels = plt.gca().get_xticklabels()
    plt.setp(labels, ha='left')
    plt.tight_layout()
    plt.savefig('Plots/conf_matrix/mixed_self/conf_matrix_all_mixed_self_model.pgf')
    plt.close()

    # filesize models
    for fs in filesizes:
        # load dic from file
        fs_results = np.load(RESULTS_FOLDER + 'conf_matrix_all_fs_' + fs + '_model.npy', allow_pickle=True).tolist()
        set_figsize()
        plt.gcf().set_size_inches(9, 7)
        sn.heatmap(fs_results, vmin=0, vmax=5000,
                   xticklabels=categories,
                   yticklabels=categories,
                   annot=True,
                   cmap='Blues',
                   fmt=".0f")
        plt.xticks(rotation=-45)
        labels = plt.gca().get_xticklabels()
        plt.setp(labels, ha='left')
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
        plt.plot(range(1, 11), fs_model_loss[:, 0], c='r', label='Train Loss')
        plt.plot(range(1, 11), fs_model_loss[:, 1], c='g', label='Validation Loss')
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(title='Loss Functions')
        plt.savefig('Plots/loss_comparison/loss_comparison_fs_' + filesize + '_model.pgf')
        plt.close()


def plot_filesize_to_target():
    # read csv
    df = pd.read_csv('filesize_log.txt', sep=',', header=None)
    df.columns = ['filesize', 'target', 'codec']
    # sort by string in codec column and reset index
    df = df.sort_values(by=['codec']).reset_index(drop=True)
    # remove / from codec column
    df['codec'] = df['codec'].str.replace('/', '')
    # scatter plot with each codec in different color with filesize as y and index as x
    set_figsize()
    # plt df with a color for each codec
    label = df['codec'].unique()
    # take the 7th element from the label array and put it at the 3rd position
    label = np.insert(label, 3, label[8])
    # remove the 8th element from the array
    label = np.delete(label, 9)

    # only take rows with target value 32
    df = df[df['target'] == 32]
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
    plt.xticks(rotation=-45)
    plt.gcf().set_size_inches(7, 6)
    labels = plt.gca().get_xticklabels()
    plt.setp(labels, ha='left')
    # Use tight_layout to ensure all elements fit within the saved area
    plt.tight_layout()
    plt.savefig('Plots/filesize_to_target.pgf')
    plt.close()


def plot_scatter_without_transfer():
    # read data from npy file
    results = np.load('results/wo_transfer_results.npy', allow_pickle=True).item()

    for filesize in results.keys():
        try:
            all_preds = np.array(results[filesize][0])
            all_labels = np.array(results[filesize][1])

            # get all unique labels
            unique_labels = np.unique(all_labels)

            plt.subplots(layout='constrained')
            set_figsize()

            markers = ['o', 'x', 's', 'D', '^', 'v', '<', '>', 'p', 'h']

            for i in range(len(unique_labels)):
                plt.scatter(all_preds[all_labels == unique_labels[i], 0],
                            all_preds[all_labels == unique_labels[i], 1], label=unique_labels[i], marker=markers[i])
            plt.legend()
            plt.gcf().set_size_inches(9, 7)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.savefig(f"Plots/plot_scatter_without_transfer_{filesize}.pgf")
            plt.close()

        except:
            continue


def plot_scatter_without_transfer_filtered():
    # read data from npy file
    results = np.load('results/wo_transfer_results.npy', allow_pickle=True).item()

    for filesize in results.keys():
        try:
            all_preds = np.array(results[filesize][0])
            all_labels = np.array(results[filesize][1])
            image_names = np.array(results[filesize][2])
            # take the first five predictions of each codec
            indices = []
            for label in np.unique(all_labels):
                indices += list(np.where(all_labels == label)[0][:5])
            indices = np.array(indices)

            all_preds = all_preds[indices]
            all_labels = all_labels[indices]
            image_names = image_names[indices]

            # create a df with the data
            df = pd.DataFrame(all_preds, columns=['PC1', 'PC2'])
            df['label'] = all_labels
            df['image'] = image_names

            fig = px.scatter(df, x='PC1', y='PC2', color='label', text='image')
            fig.update_traces(textposition='top center')
            fig.update_layout(showlegend=True)
            fig.show()

            # get all unique labels
            unique_labels = np.unique(all_labels)

            set_figsize()

            markers = ['o', 'x', 's', 'D', '^', 'v', '<', '>', 'p', 'h']
            fig, ax = plt.subplots()
            for i in range(len(unique_labels)):
                x = all_preds[all_labels == unique_labels[i], 0]
                y = all_preds[all_labels == unique_labels[i], 1]
                ax.scatter(x, y, label=unique_labels[i], marker=markers[i])
                print(x)
                for j, txt in enumerate(image_names):
                    print(j)
                    print(txt)
                    ax.annotate(txt, (x[j], y[j]))
            plt.legend()
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.savefig(f"Plots/plot_scatter_without_transfer_filtered_{filesize}.pgf")
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

    # plot_accuracy_results()
    # plot_loss_results()
    # plot_filesize_to_target()
    # plot_confusion_matrix()
    # plot_confusion_matrix_all()
    plot_scatter_without_transfer()
    plot_scatter_without_transfer_filtered()


if __name__ == '__main__':
    main()
