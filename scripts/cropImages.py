from PIL import Image
import os

path = "Images/DIV2K_train_HR/"
pathCropped = "Images/DIV2K_train_HR/Resized/"
pathVal = "Images/DIV2K_valid_HR/"
pathCroppedVal = "Images/DIV2K_valid_HR/Resized/"

def crop(outputWidth = 512, outputHeight = 512):
    pngs = []
    for file in os.listdir(path=path):
        if file.endswith(".png"):
            pngs.append(file)
    pngsVal = []
    for file in os.listdir(path=pathVal):
        if file.endswith(".png"):
            pngsVal.append(file)
    for item in pngs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            # crop the center of the image
            width, height = im.size
            left = (width - outputWidth) / 2
            top = (height - outputHeight) / 2
            right = (width + outputWidth) / 2
            bottom = (height + outputHeight) / 2
            im = im.crop((left, top, right, bottom))
            im.save(pathCropped + item[:-4] + '.png', 'PNG', quality=100)
    for item in pngsVal:
        if os.path.isfile(pathVal + item):
            im = Image.open(pathVal + item)
            # crop the center of the image
            width, height = im.size
            left = (width - outputWidth) / 2
            top = (height - outputHeight) / 2
            right = (width + outputWidth) / 2
            bottom = (height + outputHeight) / 2
            im = im.crop((left, top, right, bottom))
            im.save(pathCroppedVal + item[:-4] + '.png', 'PNG', quality=100)

if __name__ == '__main__':
    crop()