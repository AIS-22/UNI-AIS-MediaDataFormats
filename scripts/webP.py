from PIL import Image
import os
from math import floor

pathPng = '/Users/moritzlanger/DEV/FH/AIS-M2/UNI-AIS-MediaDataFormats/Images/DIV2K_train_HR/'
pathCropped = '/Users/moritzlanger/DEV/FH/AIS-M2/UNI-AIS-MediaDataFormats/Images/DIV2K_train_HR_cropped/'
pathWebP = '/Users/moritzlanger/DEV/FH/AIS-M2/UNI-AIS-MediaDataFormats/Images/DIV2K_train_HR_webP/'
imgPath = '0021'
png = '.png'
webp = '.webp'
maxFileSize = 50000

# get all png files from a folder and put the names into a list
pngs = []
for file in os.listdir(path=pathCropped):
    if file.endswith(".png"):
        pngs.append(file)
nrFiles = len(pngs)
i = 0

# convert the pngs to webp
for pngName in pngs:
    img = Image.open(pathCropped + pngName)
    qual = 0
    step = 50
    img.save(pathWebP + pngName[:-4] + webp, quality=qual)
    fileSize = os.path.getsize(pathWebP + pngName[:-4] + webp)

    # check the file size and minimize the file size
    while True:
        if fileSize > maxFileSize:
            qual = qual - step
            if qual < 0:
                break
        else:
            qual = qual + step
            if qual > 128:
                break
        img.save(pathWebP + pngName[:-4] + webp, quality=qual)
        fileSize = os.path.getsize(pathWebP + pngName[:-4] + webp)
        if step == 1 and fileSize < maxFileSize:
            while fileSize < maxFileSize:
                qual = qual + 1
                img.save(pathWebP + pngName[:-4] + webp, quality=qual)
                fileSize = os.path.getsize(pathWebP + pngName[:-4] + webp)
            break
        elif step == 1 and fileSize > maxFileSize:
            while fileSize > maxFileSize:
                qual = qual - 1
                img.save(pathWebP + pngName[:-4] + webp, quality=qual)
                fileSize = os.path.getsize(pathWebP + pngName[:-4] + webp)
            break
        step = int(floor(step / 2))

    i = i + 1
    print(pngName + ' ' + str(qual) + ' ' + str(fileSize) + ' ' + str(i) + '/' + str(nrFiles))