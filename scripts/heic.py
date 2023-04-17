import PIL
import pillow_heif
from PIL import Image
import numpy as np
import os
from math import ceil

pathPng = '/Users/moritzlanger/DEV/FH/AIS-M2/UNI-AIS-MediaDataFormats/Images/DIV2K_train_HR/'
pathCropped = '/Users/moritzlanger/DEV/FH/AIS-M2/UNI-AIS-MediaDataFormats/Images/DIV2K_train_HR_cropped/'
pathHeic = '/Users/moritzlanger/DEV/FH/AIS-M2/UNI-AIS-MediaDataFormats/Images/DIV2K_train_HR_heic/'
imgPath = '0021'
png = '.png'
heic = '.heic'
maxFileSize = 1000000
# read pngs and save as heic
pillow_heif.register_heif_opener()

# get all png files from a folder and put the names into a list
pngs = []
for file in os.listdir(path=pathPng):
    if file.endswith(".png"):
        pngs.append(file)
nrFiles = len(pngs)
i = 0

# convert the pngs to heic
for pngName in pngs:
    heifimg = pillow_heif.from_pillow(Image.open(pathPng + pngName))
    qual = 0
    step = 64
    #temp = False
    heifimg.save(pathHeic + pngName[:-4] + heic, quality=qual)
    fileSize = os.path.getsize(pathHeic + pngName[:-4] + heic)

    # check the file size and minimize the file size
    while True:
        if fileSize > maxFileSize:
            qual = qual - step
            if qual < 0:
                break
        else:
            qual = qual + step
            if qual > 100:
                break
        heifimg.save(pathHeic + pngName[:-4] + heic, quality=qual)
        fileSize = os.path.getsize(pathHeic + pngName[:-4] + heic)
        if step == 1 and fileSize < maxFileSize:
            break
        elif step == 1 and fileSize > maxFileSize:
            while fileSize > maxFileSize:
                qual = qual - 1
                heifimg.save(pathHeic + pngName[:-4] + heic, quality=qual)
                fileSize = os.path.getsize(pathHeic + pngName[:-4] + heic)
            break
        step = int(ceil(step / 2))
    print(pngName[:-4] + ' to Heif Qual: ' + str(qual) + ' Size: ' + str(fileSize)  + ' ' + str(i) + '/' + str(nrFiles))