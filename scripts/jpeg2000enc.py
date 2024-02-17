import glob
import os
import subprocess
from datetime import datetime

from filesizelogger import log_filesize

minQ = 1
maxQ = 1000
trainFolder = 'DIV2K_train_HR/'
validFolder = 'DIV2K_valid_HR/'
availableSubFolder = [trainFolder, validFolder]
usedCodec = 'JPEG2000/'

outputPrefix = 'jpeg2k_'
outputFileExtension = '.jp2'
pngExtension = '.png'


def decode_jpeg2k(enc_file, dec_file, subfolder_needed=True):
    subprocess.call('opj_decompress -o ' + dec_file + ' -i ' + enc_file,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=True)
    if subfolder_needed:
        file_size = dec_file.split('/')[-1].split('_')[-1].split('.')[0]
        dec_filesize_folder = dec_file.replace('all', file_size)
        subprocess.call('opj_decompress -o ' + dec_filesize_folder + ' -i ' + enc_file,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        shell=True)


def encode_jpeg2k(printProgress=False, maxFileSizeKb=32, useMultiCropPerImage=False):
    i = 0
    if useMultiCropPerImage:
        decodedFolder = 'Decoded_pieces/'
        croppedFolder = 'ResizedInPieces/'
    else:
        decodedFolder = 'Decoded/'
        croppedFolder = 'Resized/'
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
            original_size = os.path.getsize(image_path)
            c_rate = original_size / maxFileSizeKb
            subprocess.call('opj_compress -o ' + outputPath + ' -r ' + "{:.4f}".format(c_rate) + ' -i ' + image_path,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            shell=True)

            if printProgress:
                f_size = os.path.getsize(outputPath) / 1024
                i += 1
                print('Image: ' + file_name + ' Quality: ' + str(maxQ - q) + ' Filesize: ' + str(f_size) + ' kb' + ' Progress: ' + str(i) + '/' + str(number_of_files))
                #log into file
                log_filesize(f_size, maxFileSizeKb, usedCodec)
            
            dec_file_name = file_name.split(sep='.')[0] + '_' + str(maxFileSizeKb) + pngExtension
            dec_path = pathImagesEncoded[:-len(usedCodec)] + decodedFolder + str(maxFileSizeKb)+ "/" + usedCodec + dec_file_name
            decode_jpeg2k(outputPath, dec_path)

def encode_jpeg2k_q(image_path, decoded_path, q):
    # save image with new quality
    outputPath = 'temp' + outputFileExtension
    start_time = datetime.now()
    subprocess.call('opj_compress -o ' + outputPath + ' -r ' + str(q) + ' -i ' + image_path,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=True)
    compression_time = (datetime.now() - start_time).microseconds / 1000
    enc_size = os.path.getsize(outputPath)
    decode_jpeg2k(outputPath, decoded_path, False)
    os.system('rm ' + outputPath)
    return enc_size, compression_time


if __name__ == '__main__':
    encode_jpeg2k(printProgress=True)
