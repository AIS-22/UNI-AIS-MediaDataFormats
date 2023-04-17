import os
import glob
import numpy as np

maxFileSizeKb = 32
minQ = 0
maxQ = 51
trainFolder = 'DIV2K_train_HR/'
validFolder = 'DIV2K_valid_HR/'
usedCodec = 'BPG/'
outputPrefix = 'bpg_'
outputFileExtension = '.bpg'
pngExtension = '.png'


def encode_bpg():
    for subFolder in [trainFolder, validFolder]:
        pathImages = 'Images/' + subFolder + 'Resized/'
        pathImagesEncoded = 'Images/' + subFolder + usedCodec
        for image_path in glob.glob(pathImages + '*' + pngExtension):
            q = maxQ
            # filename is the last element of the file path also old file extension needs to be cropped
            file_name = image_path.split(sep='/')[-1].split(sep='.')[0] + outputFileExtension
            # open image and in first step use the highest available quality to store
            outputPath = pathImagesEncoded + outputPrefix + file_name
            # quality now is inverted, 0 best maxQ worst
            os.system('bpgenc -o ' + outputPath + ' -q ' + str(int(0)) + ' ' + image_path)

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
                # save image with new quality but quality now is inverted, 0 best maxQ worst, therefore maxQ-q!!
                os.system('bpgenc -o ' + outputPath + ' -q ' + str(int(maxQ - q)) + ' ' + image_path)
                if terminate_after:
                    break
                prev_q = q


if __name__ == '__main__':
    encode_bpg()
