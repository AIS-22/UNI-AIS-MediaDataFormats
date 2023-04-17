import pillow_heif
from PIL import Image
import os
import glob
import numpy as np

maxFileSizeKb = 32
minQ = 1
maxQ = 100
trainFolder = 'DIV2K_train_HR/'
validFolder = 'DIV2K_valid_HR/'
usedCodec = 'HEIC/'
outputPrefix = 'heic_'
outputFileExtension = '.heic'
pngExtension = '.png'

pillow_heif.register_heif_opener()

def encode_heic():
    i = 0
    for subFolder in [trainFolder, validFolder]:
        pathImages = 'Images/' + subFolder + 'Resized/'
        pathImagesEncoded = 'Images/' + subFolder + usedCodec
        number_of_files = len([f for f in os.listdir(pathImages) if os.path.isfile(os.path.join(pathImages, f))])
        for image_path in glob.glob(pathImages + '*' + pngExtension):
            q = maxQ
            # filename is the last element of the file path also old file extension needs to be cropped
            file_name = image_path.split(sep='/')[-1].split(sep='.')[0] + outputFileExtension
            # open image and in first step use the highest available quality to store
            outputPath = pathImagesEncoded + outputPrefix + file_name
            image = pillow_heif.from_pillow(Image.open(image_path))
            image.save(outputPath, quality=q)

            # use devide and concor to optimize computational time n*O(log(n)) complexity
            terminate_before = False
            terminate_after = False
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
                        terminate_after = True

                elif f_size < maxFileSizeKb:
                    lower_bound = prev_q
                    q += np.ceil((upper_bound - prev_q) / 2)
                    # no further optimization possible, since only one step was done
                    if q == prev_q + 1 or q == maxQ - 1:
                        # terminate before next saving, since current filesize is under threshold
                        terminate_before = True

                if terminate_before:
                    break
                # save image with new quality
                image.save(outputPath, quality=int(q))
                if terminate_after:
                    break
                prev_q = q
            i += 1
            print('Image: ' + file_name + ' Quality: ' + str(q) + ' Filesize: ' + str(f_size) + ' kb' + ' Progress: ' + str(i) + '/' + str(number_of_files))


if __name__ == '__main__':
    encode_heic()
