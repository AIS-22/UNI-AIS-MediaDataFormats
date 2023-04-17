from PIL import Image
import os

WIDTH = 512
HEIGHT = 512

path = "/Users/moritzlanger/DEV/FH/AIS-M2/UNI-AIS-MediaDataFormats/Images/DIV2K_train_HR/"
pathCropped = "/Users/moritzlanger/DEV/FH/AIS-M2/UNI-AIS-MediaDataFormats/Images/Resized/DIV2K_train_HR/"

def crop():
    pngs = []
    for file in os.listdir(path=path):
        if file.endswith(".png"):
            pngs.append(file)
    for item in pngs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            # crop the center of the image
            width, height = im.size
            left = (width - WIDTH)/2
            top = (height - HEIGHT)/2
            right = (width + WIDTH)/2
            bottom = (height + HEIGHT)/2
            im = im.crop((left, top, right, bottom))
            im.save(pathCropped + item[:-4] + '.png', 'PNG', quality=100)

crop()