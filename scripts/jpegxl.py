import os
import subprocess
import glob
import numpy as np
from imagecodecs import imread, imwrite
from filesizelogger import log_filesize


minQ = 1
maxQ = 100
trainFolder = 'DIV2K_train_HR/'
validFolder = 'DIV2K_valid_HR/'
availableSubFolder = [trainFolder, validFolder]
usedCodec = 'JPEG_XL/'

outputPrefix = 'jpegxl_'
outputFileExtension = '.jxl'
pngExtension = '.png'


def decode_jpgxl(enc_file, dec_file):
    image = imread(enc_file)
    imwrite(dec_file, image, 'png')
    file_size = dec_file.split('/')[-1].split('_')[-1].split('.')[0]
    dec_filesize_folder = dec_file.replace('all', file_size)
    imwrite(dec_filesize_folder, image, 'png')

def encode_jpgxl(printProgress=False, maxFileSizeKb = 32, useMultiCropPerImage = True):
    i = 0
    if useMultiCropPerImage:
        decodedFolder = 'Decoded_pieces/'
        croppedFolder = 'ResizedInPieces/'
    else:
        decodedFolder = 'Decoded/'
        croppedFolder = 'Resized/'
    #number_of_files = len(glob.glob('Images/' + '*/' + '*' + pngExtension))
    number_of_files = len(glob.glob('Images/*/' + croppedFolder + '/*.png'))

    for subFolder in availableSubFolder:
        pathImages = 'Images/' + subFolder + croppedFolder
        pathImagesEncoded = 'Images/' + subFolder + usedCodec
        for image_path in glob.glob(pathImages + '*' + pngExtension):
            q = maxQ
            # filename is the last element of the file path also old file extension needs to be cropped
            file_name = outputPrefix + image_path.split(sep='/')[-1].split(sep='.')[0] + outputFileExtension
            # open image and in first step use the highest available quality to store
            outputPath = pathImagesEncoded + file_name

            #encode image with quality q
            subprocess.call(['cjxl', image_path, outputPath, '--quiet', '-q', str(q)])
            

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
                subprocess.call(['cjxl', image_path, outputPath, '--quiet', '-q', str(q)])
                if terminate:
                    if os.path.getsize(outputPath) / 1024 > maxFileSizeKb and q > minQ:
                        #ML: decrease quality by 1 to get under threshold again and to set q to the last valid value
                        q = q - 1
                        subprocess.call(['cjxl', image_path, outputPath, '--quiet', '-q', str(q)])
                    break
                prev_q = q

            if printProgress:
                f_size = os.path.getsize(outputPath) / 1024
                i += 1
                print('Image: ' + file_name + ' Quality: ' + str(q) + ' Filesize: ' + str(f_size) + ' kb' + ' Progress: ' + str(i) + '/' + str(number_of_files))
                #log into file
                log_filesize(f_size, maxFileSizeKb, usedCodec)
                
            dec_file_name = file_name.split(sep='.')[0] + '_' + str(maxFileSizeKb) + pngExtension
            dec_path = pathImagesEncoded[:-len(usedCodec)] + decodedFolder + str(maxFileSizeKb)+ "/" + usedCodec + dec_file_name
            decode_jpgxl(outputPath, dec_path)

def encode_jxl_q(image_path, decoded_path, q):
    # save image with new quality
    outputPath = 'temp' + outputFileExtension
    subprocess.call(['cjxl', image_path, outputPath, '--quiet', '-q', str(q)])
    enc_size = os.path.getsize(outputPath)
    decode_jpgxl(outputPath, decoded_path)
    os.system('rm ' + outputPath)
    return enc_size


if __name__ == '__main__':
    encode_jpgxl(printProgress=True)
