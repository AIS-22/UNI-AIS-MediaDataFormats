import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import avifenc
import bpgenc
import heicEnc
import jxrenc
import webP
import jpegxl

np.random.seed(86)


def calculate_mse(image1, image2):
    error = np.square(np.subtract(image1, image2))
    return np.mean(error)


def calc_psnr(original_image, decoded_image):
    orig = Image.open(original_image)
    dec = Image.open(decoded_image)
    mse = calculate_mse(orig, dec)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(np.array(orig).max() / np.sqrt(mse))


def measure_quality():
    os.system('mkdir Quality')
    os.system('mkdir Quality/Decoded')

    files = glob.glob('Images/' + '*/Resized/' + '*.png')
    n_files = len(files)
    # pick 5 images for quality measurement
    n_images = 5
    indices = np.floor(np.random.uniform(low=0, high=n_files - 1, size=n_images)).astype(int)
    image_paths = [files[i] for i in indices]
    # make 50 quality steps (2 per step)
    steps = 50
    qualities = np.linspace(1, 100, steps).astype(int)
    # qualities for jxr since range can be 0-10_000
    qualities_jxr = np.linspace(0, 10_000, steps).astype(int)
    codecs = ['avif', 'webP', 'bpg', 'heic', 'jxl', 'jxr']

    codec_dictionary = {
        'avif': avifenc.encode_avif_q,
        'webP': webP.encode_webP_q,
        'bpg': bpgenc.encode_bpg_q,
        'heic': heicEnc.encode_heic_q,
        'jxl': jpegxl.encode_jxl_q,
        'jxr': jxrenc.encode_jxr_q
    }

    for codec in codecs:
        mean_crates = np.zeros(len(qualities))
        mean_psnr = np.zeros(len(qualities))
        codec_is_jxr = codec == 'jxr'
        for x, q in enumerate(qualities_jxr if codec_is_jxr else qualities):
            c_rates = np.zeros(n_images).astype(float)
            psnr = np.zeros(n_images).astype(float)
            for i, file_path in enumerate(image_paths):
                file_name = file_path.split(sep='/')[-1].split(sep='.')[0] + '_' + str(q) + '_' + codec + '.png'
                decoded_path = 'Quality/Decoded/' + file_name
                enc_size = codec_dictionary[codec](file_path, decoded_path, q)
                c_rates[i] = os.path.getsize(file_path) / enc_size
                psnr[i] = calc_psnr(file_path, decoded_path)
            mean_crates[x] = np.mean(c_rates)
            mean_psnr[x] = np.mean(psnr)
        plt.plot(mean_crates, mean_psnr, label=codec)

    plt.legend()
    plt.xlabel('Compression Rate')
    plt.ylabel('PSNR')
    plt.title('Comparison of the Codecs Regarding PSNR')
    plt.savefig('Plots/psnr.png')


if __name__ == '__main__':
    measure_quality()
