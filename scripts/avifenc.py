# %pip install pillow-avif-plugin
import pillow_avif
from PIL import Image
import os
import glob
import numpy as np
from filesizelogger import log_filesize


minQ = 1
maxQ = 100
trainFolder = 'DIV2K_train_HR/'
validFolder = 'DIV2K_valid_HR/'
availableSubFolder = [trainFolder, validFolder]
usedCodec = 'AVIF/'
decodedFolder = 'Decoded/'
outputPrefix = 'avif_'
outputFileExtension = '.avif'
pngExtension = '.png'


def decode_avif(enc_file, dec_file):
    image = Image.open(enc_file)
    image.save(dec_file, quality=100)
    file_size = dec_file.split('/')[-1].split('_')[-1].split('.')[0]
    dec_filesize_folder = dec_file.replace('all', file_size)
    image.save(dec_filesize_folder, quality=100)

def encode_avif(printProgress=False, maxFileSizeKb = 32):
    i = 0
    #number_of_files = len(glob.glob('Images/' + '*/' + '*' + pngExtension))
    number_of_files = len(glob.glob('Images/*/' + croppedFolder + '*.png'))
    for subFolder in availableSubFolder:
        pathImages = 'Images/' + subFolder + croppedFolder
        pathImagesEncoded = 'Images/' + subFolder + usedCodec
        for image_path in glob.glob(pathImages + '*' + pngExtension):
            q = maxQ
            # filename is the last element of the file path also old file extension needs to be cropped
            file_name = outputPrefix + image_path.split(sep='/')[-1].split(sep='.')[0] + outputFileExtension
            # open image and in first step use the highest available quality to store
            outputPath = pathImagesEncoded + file_name
            image = Image.open(image_path)
            image.save(outputPath, quality=q)

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

                # save image with new quality
                image.save(outputPath, quality=int(q))
                if terminate:
                    # there was a rounding error caused by np.ceil() so just one more optimization step is needed
                    if os.path.getsize(outputPath) / 1024 > maxFileSizeKb and q > minQ:
                        q = q - 1
                        image.save(outputPath, quality=int(q))
                    break
                prev_q = q

            if printProgress:
                f_size = os.path.getsize(outputPath) / 1024
                i += 1
                print('Image: ' + file_name + ' Quality: ' + str(q) + ' Filesize: ' + str(f_size) + ' kb' + ' Progress: ' + str(i) + '/' + str(number_of_files))
                #log into file
                log_filesize(f_size, maxFileSizeKb, usedCodec)

            dec_file_name = file_name.split(sep='.')[0] + '_' + str(maxFileSizeKb) + pngExtension
            dec_path = pathImagesEncoded[:-len(usedCodec)] + decodedFolder + str(maxFileSizeKb)+ "/"   + usedCodec + dec_file_name
            decode_avif(outputPath, dec_path)

def encode_avif_q(image_path, decoded_path, q):
    image = Image.open(image_path)
    # save image with new quality
    outputPath = 'temp' + outputFileExtension
    image.save(outputPath, quality=int(q))
    enc_size = os.path.getsize(outputPath)
    decode_avif(outputPath, decoded_path)
    os.system('rm ' + outputPath)
    return enc_size


if __name__ == '__main__':
    encode_avif(printProgress=True)
