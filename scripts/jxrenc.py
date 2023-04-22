import os
import glob
import numpy as np
from PIL import Image

minQ = 0
maxQ = 255
trainFolder = 'DIV2K_train_HR/'
validFolder = 'DIV2K_valid_HR/'
availableSubFolder = [trainFolder, validFolder]
usedCodec = 'JPEG_XR/'
decodedFolder = 'Decoded/'
outputPrefix = 'jxr_'
outputFileExtension = '.jxr'
pngExtension = '.png'
tifFileExtension = '.tif'


def decode_jxr(enc_file, dec_file):
    os.system('./jpegxr -o ' + dec_file + ' ' + enc_file)
    # convert tif to png
    im = Image.open(dec_file)
    png_file_name = dec_file.split(sep='.')[0] + pngExtension
    im.save(png_file_name, quality=100)
    # remove the created tif image
    os.system('rm ' + dec_file)

def encode_jxr(printProgress=False, maxFileSizeKb = 32):
    i = 0
    number_of_files = len(glob.glob('Images/' + '*/' + '*' + pngExtension))
    for subFolder in availableSubFolder:
        pathImages = 'Images/' + subFolder + 'Resized/'
        pathImagesEncoded = 'Images/' + subFolder + usedCodec
        for image_path in glob.glob(pathImages + '*' + pngExtension):
            q = maxQ
            # filename is the last element of the file path also old file extension needs to be cropped
            file_name = outputPrefix + image_path.split(sep='/')[-1].split(sep='.')[0] + outputFileExtension
            tif_path = image_path.split(sep='.')[0] + tifFileExtension
            # open image and in first step use the highest available quality to store
            outputPath = pathImagesEncoded + file_name

            # convert png to tif format, otherwise jxr is not working
            im = Image.open(image_path)
            im.save(tif_path, quality=100)

            # quality now is inverted, 0 lossless maxQ max compression
            os.system('./jpegxr -c -q ' + str(int(0)) + ' -o ' + outputPath + ' ' + tif_path)

            # use devide and concor to optimize computational time n*O(log(n)) complexity
            terminate = False
            prev_q = q
            upper_bound = maxQ
            lower_bound = 0
            while True:
                f_size = os.path.getsize(outputPath) / 1024
                # filesize can´t be optimized, since max. quality is already under threshold
                if q == maxQ and f_size <= maxFileSizeKb:
                    break

                if f_size > maxFileSizeKb:
                    upper_bound = prev_q
                    q -= np.ceil((upper_bound - lower_bound) / 2)
                    # no further optimization possible, since only one step was done
                    if prev_q == minQ or (q == prev_q - 1):
                        # terminate after next saving, since current filesize is above threshold
                        terminate = True

                elif f_size < maxFileSizeKb:
                    lower_bound = prev_q
                    q += np.ceil((upper_bound - prev_q) / 2)
                    # no further optimization possible, since only one step was done
                    if q == prev_q + 1 or q == maxQ:
                        # terminate before next saving, since current filesize is under threshold
                        terminate = True

                elif f_size == maxFileSizeKb:
                    break

                # save image with new quality but quality now is inverted, 0 lossless maxQ max compression, therefore maxQ-q!!
                os.system('./jpegxr -c -q ' + str(int(maxQ - q)) + ' -o ' + outputPath + ' ' + tif_path)
                if terminate:
                    # there was a rounding error caused by np.ceil() so just one more optimization step is needed
                    if os.path.getsize(outputPath) / 1024 > maxFileSizeKb:
                        q = q - 1
                        os.system('./jpegxr -c -q ' + str(int(maxQ - q)) + ' -o ' + outputPath + ' ' + tif_path)
                    # remove the created tif image
                    os.system('rm ' + tif_path)
                    break
                prev_q = q

            if printProgress:
                f_size = os.path.getsize(outputPath) / 1024
                i += 1
                print('Image: ' + file_name + ' Quality: ' + str(maxQ - q) + ' Filesize: ' + str(f_size) + ' kb' + ' Progress: ' + str(i) + '/' + str(number_of_files))

            dec_file_name = file_name.split(sep='.')[0] + tifFileExtension
            dec_path = pathImagesEncoded + decodedFolder + dec_file_name
            decode_jxr(outputPath, dec_path)


if __name__ == '__main__':
    encode_jxr(printProgress=True)
