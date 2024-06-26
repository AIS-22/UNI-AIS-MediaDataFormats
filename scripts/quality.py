import webP
import jxrenc_2
import jxrenc_1
import jxrenc_0
import jpegxl
import jpegenc
import jpeg2000enc
import heicEnc
import bpgenc
import avifenc
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import glob
import os

### DISCLAIMER ###
# To import pillow_heif before cv2 is necessary to avoid a bug in the library
# affected devices: macOS on Apple Silicon
# https://github.com/bigcat88/pillow_heif/issues/89
import pillow_heif

pillow_heif.register_heif_opener()
### END DISCLAIMER ###


# Make matplotlib use latex for font rendering
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


np.random.seed(86)
codec_mapping = {
    'avif': "AVIF",
    'webP': "WEBP",
    'bpg': "BPG",
    'heic': "HEIC",
    'jxl': "JPEG XL",
    'jxr_0': "JPEG XR_{0}",
    'jxr_1': "JPEG XR_{1}",
    'jxr_2': "JPEG XR_{2}",
    'jpeg': "JPEG",
    'jpeg2000': "JPEG 2000"
}
jxr_parameter = ["q (0.0-1.0)", "q (2-255)"]


def calculate_mse(image1, image2):
    error = np.square(np.subtract(image1, image2))
    return np.mean(error)


def calc_psnr(original_image_path, decoded_image_path):
    orig = Image.open(original_image_path)
    dec = Image.open(decoded_image_path)
    mse = calculate_mse(orig, dec)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(np.array(orig).max() / np.sqrt(mse))


def calc_ssim(orig_image_path, dec_image_path):
    orig_image = cv2.imread(orig_image_path)
    dec_image = cv2.imread(dec_image_path)

    # Validate arguments
    assert orig_image.dtype == np.uint8 and dec_image.dtype == np.uint8, 'Images must be 8-bit'
    assert orig_image.shape == dec_image.shape, 'Images must have the same size'

    # Constants to stabilize the division with weak denominator
    k1 = 0.01
    k2 = 0.03
    L = 2 ** 8 - 1  # 8-bit dynamic range of pixel-values
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    # Filtering options
    opts = {'ksize': (11, 11), 'sigmaX': 1.5, 'sigmaY': 1.5}

    # Work in floating-point precision
    I1 = orig_image.astype(float)
    I2 = dec_image.astype(float)

    # Mean
    mu1 = cv2.GaussianBlur(I1, **opts)
    mu2 = cv2.GaussianBlur(I2, **opts)

    # Variance
    sigma1_2 = cv2.GaussianBlur(I1 ** 2, **opts) - mu1 ** 2
    sigma2_2 = cv2.GaussianBlur(I2 ** 2, **opts) - mu2 ** 2

    # Covariance
    sigma12 = cv2.GaussianBlur(I1 * I2, **opts) - mu1 * mu2

    # SSIM index
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_2 + sigma2_2 + C2)
    map_ssim = num / den
    val_ssim = np.mean(map_ssim)

    return val_ssim


def get_empty_result_dict(len_qualities):
    return dict(avif=np.zeros((4, len_qualities)).astype(float),
                webP=np.zeros((4, len_qualities)).astype(float),
                bpg=np.zeros((4, len_qualities)).astype(float),
                heic=np.zeros((4, len_qualities)).astype(float),
                jxl=np.zeros((4, len_qualities)).astype(float),
                jxr_0=np.zeros((4, len_qualities)).astype(float),
                jxr_1=np.zeros((4, len_qualities)).astype(float),
                jxr_2=np.zeros((4, len_qualities)).astype(float),
                jpeg=np.zeros((4, len_qualities)).astype(float),
                jpeg2000=np.zeros((4, len_qualities)).astype(float))


def measure_quality(useMultiCropPerImage=False):
    if useMultiCropPerImage:
        croppedFolder = 'ResizedInPieces/'
    else:
        croppedFolder = 'Resized/'
    # get all images
    files = glob.glob('Images/' + '*/' + croppedFolder + '*.png')
    n_files = len(files)
    # pick 5 images for quality measurement
    n_images = 5
    indices = np.floor(np.random.uniform(low=0, high=n_files - 1, size=n_images)).astype(int)
    image_paths = [files[i] for i in indices]
    # make 50 quality steps
    steps = 50
    qualities = np.linspace(1, 100, steps).astype(int)
    # qualities for jxr since range can be 0-1 for quality, whereas from 2-255 the quantisation gets changed
    qualities_jxr = 0.
    qualities_jxr = np.append(qualities_jxr, np.linspace(0.0001, 1, int(steps / 2) - 1).astype(float))
    qualities_jxr = np.append(qualities_jxr, np.linspace(2, 255, int(steps / 2)).astype(int))
    qualities_j2k = np.concatenate(
        (np.linspace(1, 100, int(steps / 2)).astype(int), np.linspace(104, 1000, int(steps / 2)).astype(int)))
    len_qualities = len(qualities)

    codec_dictionary = {
        'avif': avifenc.encode_avif_q,
        'webP': webP.encode_webP_q,
        'bpg': bpgenc.encode_bpg_q,
        'heic': heicEnc.encode_heic_q,
        'jxl': jpegxl.encode_jxl_q,
        'jxr_0': jxrenc_0.encode_jxr_q,
        'jxr_1': jxrenc_1.encode_jxr_q,
        'jxr_2': jxrenc_2.encode_jxr_q,
        'jpeg': jpegenc.encode_jpeg_q,
        'jpeg2000': jpeg2000enc.encode_jpeg2k_q
    }

    results = get_empty_result_dict(len_qualities)

    for codec in codec_dictionary.keys():
        mean_crates = np.zeros(len_qualities)
        mean_psnr = np.zeros(len_qualities)
        mean_ssim = np.zeros(len_qualities)
        mean_time = np.zeros(len_qualities)
        codec_is_jxr = 'jxr' in codec
        codec_is_j2k = codec == 'jpeg2000'
        for x, q in enumerate(qualities_jxr if codec_is_jxr else qualities_j2k if codec_is_j2k else qualities):
            c_rates = np.zeros(n_images).astype(float)
            psnr = np.zeros(n_images).astype(float)
            ssim = np.zeros(n_images).astype(float)
            time = np.zeros(n_images).astype(float)
            for i, file_path in enumerate(image_paths):
                q_string = str(q).replace('.', '_')
                file_name = file_path.split(sep='/')[-1].split(sep='.')[0] + '_' + q_string + '_' + codec + '.png'
                decoded_path = 'Quality/Decoded/' + file_name
                enc_size, compression_time = codec_dictionary[codec](file_path, decoded_path, q)
                time[i] = compression_time
                c_rates[i] = os.path.getsize(file_path) / enc_size
                psnr[i] = calc_psnr(file_path, decoded_path)
                ssim[i] = calc_ssim(file_path, decoded_path)
            mean_time[x] = np.mean(time)
            mean_crates[x] = np.mean(c_rates)
            mean_psnr[x] = np.mean(psnr)
            mean_ssim[x] = np.mean(ssim)
        results[codec][0] = mean_crates
        results[codec][1] = mean_psnr
        results[codec][2] = mean_ssim
        results[codec][3] = mean_time

    # store the results in a file
    np.save('results/results_quality.npy', results)


def _plot_results(codecs, results, metric, save_path, x_lim, y_lim_psnr, y_lim_ssim):
    set_figsize(figsize=(5, 5))

    cmap = plt.get_cmap('tab20')
    index = 0
    line_dict = {
        'avif': (0, (1, 1)),
        'webP': (0, (7, 4, 1, 4)),
        'bpg':  (0, (7, 1, 1, 5)),
        'heic': (0, (7, 5, 1, 1)),
        'jxl': '-',
        'jpeg': (0, (5, 1)),
        'jpeg2000': (0, (5, 6))
    }
    line_dict_jxr = {
        ('jxr_0', 'q (0.0-1.0)'): (0, (7, 5, 1, 1, 1, 5)),
        ('jxr_0', 'q (2-255)'): (0, (7, 4, 1, 1, 1, 1, 1, 4)),
        ('jxr_1', 'q (0.0-1.0)'): (0, (7, 1, 1, 1, 1, 9)),
        ('jxr_1', 'q (2-255)'): (0, (7, 1, 1, 1, 1, 1, 1, 7)),
        ('jxr_2', 'q (0.0-1.0)'): (0, (7, 9, 1, 1, 1, 1)),
        ('jxr_2', 'q (2-255)'): (0, (7, 7, 1, 1, 1, 1, 1, 1)),
    }
    for codec in codecs:
        codec_results = results[codec]
        mean_crates = codec_results[0]
        mean_metric = codec_results[1]
        if "jxr" in codec:
            for measure in jxr_parameter:
                metric_subset = mean_metric[:25] if "." in measure else mean_metric[25:]
                crates_subset = mean_crates[:25] if "." in measure else mean_crates[25:]
                plt.plot(crates_subset, metric_subset,
                         label=codec_mapping[codec] + " " + measure, color=cmap(index), linestyle=line_dict_jxr[(codec, measure)])
                index += 1
            continue
        plt.plot(mean_crates, mean_metric, label=codec_mapping[codec], color=cmap(index), linestyle=line_dict[codec])
        index += 1

    plt.legend(title='Codecs (quality parameter)', loc='upper left', bbox_to_anchor=(1, 1))
    plt.gcf().set_size_inches(9, 5)
    # Use tight_layout to ensure all elements fit within the saved area
    plt.tight_layout()
    if metric == 'PSNR':
        plt.ylim(y_lim_psnr)
    elif metric == 'SSIM' and y_lim_ssim:
        plt.ylim(y_lim_ssim)
    plt.xlim(x_lim)
    plt.xlabel('Compression Rate')
    plt.ylabel(metric)
    plt.savefig(save_path + '.pgf')
    plt.close()


def plot_enc_time_bar(time_dict):
    set_figsize()

    cmap = plt.get_cmap('tab20')
    mean_enc_time = []
    used_codec_strings = []

    for codec in time_dict.keys():
        if "jxr" in codec:
            for measure in jxr_parameter:
                used_codec_strings.append(codec_mapping[codec] + "  " + measure)
                mean_enc_time.append(time_dict[codec][1].mean())
            continue
        used_codec_strings.append(codec_mapping[codec])
        mean_enc_time.append(time_dict[codec][1].mean())

    bars = plt.bar(used_codec_strings, mean_enc_time, color=cmap(np.arange(len(used_codec_strings))))
    plt.legend(bars, used_codec_strings, loc='upper left', bbox_to_anchor=(1, 1))
    plt.gcf().set_size_inches(15, 10)
    plt.xticks([])
    plt.ylabel('Mean Time (ms)')
    plt.savefig('Plots/encoding_time_comparison.pgf', bbox_inches='tight')
    plt.close()


def plot_results():
    # load dic from file
    results = np.load('results/results_quality.npy', allow_pickle=True).item()
    codecs = results.keys()

    psnr_sub_dict = {key: [results[key][0], results[key][1]] for key in codecs}
    ssim_sub_dict = {key: [results[key][0], results[key][2]] for key in codecs}
    time_sub_dict = {key: [results[key][0], results[key][3]] for key in codecs}

    _plot_results(codecs, psnr_sub_dict, 'PSNR', 'Plots/psnr', (4, 100), (28, 45), None)
    _plot_results(codecs, ssim_sub_dict, 'SSIM', 'Plots/ssim', (4, 100), None, None)
    _plot_results(codecs, psnr_sub_dict, 'PSNR', 'Plots/psnr_adapted', (10, 100), (28, 38), None)
    _plot_results(codecs, ssim_sub_dict, 'SSIM', 'Plots/ssim_adapted', (10, 100), None, (0.45, 1))
    _plot_results(codecs, time_sub_dict, 'Time (ms)', 'Plots/enc_time', (4, 100), None, None)
    plot_enc_time_bar(time_sub_dict)


def set_figsize(figsize=(7, 5)):
    plt.figure(figsize=figsize)
    plt.rcParams['font.size'] = 15


if __name__ == '__main__':
   # measure_quality()
    plot_results()
