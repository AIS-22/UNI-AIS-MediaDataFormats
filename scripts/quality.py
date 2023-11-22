import glob
import os

### DISCLAIMER ###
# To import pillow_heif before cv2 is necessary to avoid a bug in the library
# affected devices: macOS on Apple Silicon
# https://github.com/bigcat88/pillow_heif/issues/89
import pillow_heif

pillow_heif.register_heif_opener()
### END DISCLAIMER ###

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import avifenc
import bpgenc
import heicEnc
import jpeg2000enc
import jpegenc
import jpegxl
import jxrenc_0
import jxrenc_1
import jxrenc_2
import webP

np.random.seed(86)


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


def _plot_results(codecs, results, metric, save_path):
    plt.rc('font', size=23)
    plt.figure(figsize=(13, 13))

    for codec in codecs:
        codec_results = results[codec]
        mean_crates = codec_results[0]
        mean_metric = codec_results[1]
        plt.plot(mean_crates, mean_metric, label=codec)

    plt.legend()
    if metric == 'PSNR':
        plt.ylim((28, 45))
    plt.xlim((4, 100))
    plt.xlabel('Compression Rate')
    plt.ylabel(metric)
    plt.savefig(save_path)
    plt.close()


def plot_results():
    # load dic from file
    results = np.load('results/results_quality.npy', allow_pickle=True).item()
    codecs = results.keys()

    psnr_sub_dict = {key: [results[key][0], results[key][1]] for key in codecs}
    ssim_sub_dict = {key: [results[key][0], results[key][2]] for key in codecs}
    time_sub_dict = {key: [results[key][0], results[key][3]] for key in codecs}

    _plot_results(codecs, psnr_sub_dict, 'PSNR', 'Plots/psnr.png')
    _plot_results(codecs, ssim_sub_dict, 'SSIM', 'Plots/ssim.png')
    _plot_results(codecs, time_sub_dict, 'Time (ms)', 'Plots/enc_time.png')


if __name__ == '__main__':
    measure_quality()
    plot_results()
    print('Quality measurement successful.')
