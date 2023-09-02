import os
import glob
import numpy as np
from PIL import Image

minQ = 1
# to increase the floating point precision this high value is needed (q needs to be in range 0-1)
maxQ = 10_000
maxQuantisation = 255
trainFolder = 'DIV2K_train_HR/'
validFolder = 'DIV2K_valid_HR/'
availableSubFolder = [trainFolder, validFolder]
usedCodec = 'JPEG_XR_0/'
overlapParameter = '-l 0'
decodedFolder = 'Decoded_pieces/'
outputPrefix = 'jxr_0_'
outputFileExtension = '.jxr'
pngExtension = '.png'
tifFileExtension = '.tif'


def decode_jxr(enc_file, dec_file):
    os.system('JxrDecApp -i ' + enc_file + ' -o ' + dec_file)
    # convert tif to png
    im = Image.open(dec_file)
    png_file_name = dec_file.split(sep='.')[0] + pngExtension
    im.save(png_file_name, quality=100)
    file_size = png_file_name.split('/')[-1].split('_')[-1].split('.')[0]
    dec_filesize_folder = png_file_name.replace('all', file_size)
    im.save(dec_filesize_folder, quality=100)
    # remove the created tif image
    os.system('rm ' + dec_file)

def _determine_q(max_q, max_file_size_kb, output_path, tif_path):
    # use devide and concor to optimize computational time n*O(log(n)) complexity
    is_quantization = max_q == maxQuantisation
    terminate = False
    q = max_q
    prev_q = q
    upper_bound = max_q
    lower_bound = 0

    os.system('JxrEncApp -q ' + str(1) + ' -o ' + output_path + ' -i ' + tif_path + ' ' + overlapParameter)

    while True:
        f_size = os.path.getsize(output_path) / 1024
        # filesize can´t be optimized, since max. quality is already under threshold
        if q == max_q and f_size <= max_file_size_kb:
            return max_q, False
            #break

        if f_size > max_file_size_kb:
            upper_bound = prev_q
            q -= np.ceil((upper_bound - lower_bound) / 2)
            # no further optimization possible, since only one step was done
            if prev_q == minQ or (q == prev_q - 1):
                # terminate after next saving, since current filesize is above threshold
                terminate = True

        elif f_size < max_file_size_kb:
            lower_bound = prev_q
            q += np.ceil((upper_bound - prev_q) / 2)
            # no further optimization possible, since only one step was done
            if q == prev_q + 1 or q == max_q:
                # terminate before next saving, since current filesize is under threshold
                terminate = True

        elif f_size == max_file_size_kb:
            return ((q / maxQ), False) if not is_quantization else (int(maxQuantisation - q), True)

        q_str = str(q / max_q) if not is_quantization else str(int(max_q - q))
        os.system('JxrEncApp -q ' + q_str + ' -o ' + output_path + ' -i ' + tif_path + ' ' + overlapParameter)
        if terminate:
            # there was a rounding error caused by np.ceil() so just one more optimization step is needed
            if os.path.getsize(output_path) / 1024 > max_file_size_kb and q > 0:
                q = q - 1
                q_str = str(q / max_q) if not is_quantization else str(int(max_q - q))
                os.system('JxrEncApp -q ' + q_str + ' -o ' + output_path + ' -i ' + tif_path + ' ' + overlapParameter)
            if not is_quantization and os.path.getsize(output_path) / 1024 > max_file_size_kb and q == 0:
                # recursive call sec step determine quantization since quality parameter is not able to achieve file size
                q, is_quantization = _determine_q(maxQuantisation, max_file_size_kb, output_path, tif_path)
            return ((q / maxQ), False) if not is_quantization else (int(maxQuantisation - q), True)
        prev_q = q

def encode_jxr(printProgress=False, maxFileSizeKb = 32):
    i = 0
    number_of_files = len(glob.glob('Images/' + '*/' + '*' + pngExtension))
    for subFolder in availableSubFolder:
        pathImages = 'Images/' + subFolder + 'ResizedInPieces/'
        pathImagesEncoded = 'Images/' + subFolder + usedCodec
        for image_path in glob.glob(pathImages + '*' + pngExtension):
            # filename is the last element of the file path also old file extension needs to be cropped
            
            file_name = outputPrefix + image_path.split(sep='/')[-1].split(sep='.')[0] + outputFileExtension
            tif_path = image_path.split(sep='.')[0] + tifFileExtension
            # open image and in first step use the highest available quality to store
            outputPath = pathImagesEncoded + file_name
            # print("file_name = " + file_name)
            # print("tif path = " + tif_path)
            # print("output path = " + outputPath)
            # convert png to tif format, otherwise jxr is not working
            im = Image.open(image_path)
            im.save(tif_path, quality=100)

            q, _ = _determine_q(maxQ, maxFileSizeKb, outputPath, tif_path)
            # remove the created tif image
            os.system('rm ' + tif_path)

            if printProgress:
                f_size = os.path.getsize(outputPath) / 1024
                i += 1
                print('Image: ' + file_name + ' Quality: ' + str(q) + ' Filesize: ' + str(f_size) + ' kb' + ' Progress: ' + str(i) + '/' + str(number_of_files))

            dec_file_name = file_name.split(sep='.')[0] + '_' + str(maxFileSizeKb) + tifFileExtension
            dec_path = pathImagesEncoded[:-len(usedCodec)] + decodedFolder + str(maxFileSizeKb)+ "/" + usedCodec + dec_file_name
            decode_jxr(outputPath, dec_path)

def encode_jxr_q(image_path, decoded_path, q):
    outputPath = 'temp' + outputFileExtension
    tif_path = decoded_path.split(sep='.')[0] + '.tif'
    im = Image.open(image_path)
    im.save(tif_path, quality=100)
    q_string = str(q) if q <= 1 else str(int(q))
    os.system('JxrEncApp -q ' + q_string + ' -o ' + outputPath + ' -i ' + tif_path + ' ' + overlapParameter)
    enc_size = os.path.getsize(outputPath)
    decode_jxr(outputPath, tif_path)
    os.system('rm temp*')
    return enc_size


if __name__ == '__main__':
    encode_jxr(printProgress=True)
