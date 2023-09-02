from PIL import Image
import os
import sys

path = "Images/DIV2K_train_HR/"
pathCropped = "Images/DIV2K_train_HR/"
pathVal = "Images/DIV2K_valid_HR/"
pathCroppedVal = "Images/DIV2K_valid_HR/"

def crop_center(outputWidth = 512, outputHeight = 512, pngs=[], pngsVal=[]):
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
    return

def crop_leftTop(outputWidth = 512, outputHeight = 512, pngs=[], pngsVal=[]):
    for item in pngs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            # crop left top of the image
            width, height = im.size
            center_x = width/2
            center_y = height/2
            # if image is to small to cut from the center, image is cut by the image corner
            if center_x < outputWidth or center_y < outputHeight:
                left = 0
                top = 0
                right = outputWidth
                bottom = outputHeight
            else:
                left = center_x - outputWidth
                top = center_y - outputHeight
                right = center_x 
                bottom = center_y

            im = im.crop((left, top, right, bottom))
            im.save(pathCropped + item[:-4] + '_1.png', 'PNG', quality=100)

    for item in pngsVal:
        if os.path.isfile(pathVal + item):
            im = Image.open(pathVal + item)
            # crop left top of the image
            width, height = im.size
            center_x = width/2
            center_y = height/2
            # if image is to small to cut from the center, image is cut by the image corner
            if center_x < outputWidth or center_y < outputHeight:
                left = 0
                top = 0
                right = outputWidth
                bottom = outputHeight
            else:
                left = center_x - outputWidth
                top = center_y - outputHeight
                right = center_x 
                bottom = center_y
            im = im.crop((left, top, right, bottom))
            im.save(pathCroppedVal + item[:-4] + '_1.png', 'PNG', quality=100)
    return

def crop_rightTop(outputWidth = 512, outputHeight = 512, pngs=[], pngsVal=[]):
    for item in pngs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            # crop left top of the image
            width, height = im.size
            center_x = width/2
            center_y = height/2
            # if image is to small to cut from the center, image is cut by the image corner
            if center_x < outputWidth or center_y < outputHeight:
                left = width - outputWidth
                top = 0
                right = width
                bottom = outputHeight
            else:
                left = center_x
                top = center_y - outputHeight
                right = center_x + outputWidth
                bottom = center_y

            im = im.crop((left, top, right, bottom))
            im.save(pathCropped + item[:-4] + '_2.png', 'PNG', quality=100)

    for item in pngsVal:
        if os.path.isfile(pathVal + item):
            im = Image.open(pathVal + item)
            # crop left top of the image
            width, height = im.size
            center_x = width/2
            center_y = height/2
            # if image is to small to cut from the center, image is cut by the image corner
            if center_x < outputWidth or center_y < outputHeight:
                left = width - outputWidth
                top = 0
                right = width
                bottom = outputHeight
            else:
                left = center_x
                top = center_y - outputHeight
                right = center_x + outputWidth
                bottom = center_y

            im = im.crop((left, top, right, bottom))
            im.save(pathCroppedVal + item[:-4] + '_2.png', 'PNG', quality=100)
    return

def crop_leftBottom(outputWidth = 512, outputHeight = 512, pngs=[], pngsVal=[]):
    for item in pngs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            # crop left top of the image
            width, height = im.size
            center_x = width/2
            center_y = height/2
            # if image is to small to cut from the center, image is cut by the image corner
            if center_x < outputWidth or center_y < outputHeight:
                left = 0
                top = height - outputHeight
                right = outputWidth
                bottom = height
            else:
                left = center_x - outputWidth
                top = center_y
                right = center_x 
                bottom = center_y + outputHeight

            im = im.crop((left, top, right, bottom))
            im.save(pathCropped + item[:-4] + '_3.png', 'PNG', quality=100)
            

    for item in pngsVal:
        if os.path.isfile(pathVal + item):
            im = Image.open(pathVal + item)
            # crop left top of the image
            width, height = im.size
            center_x = width/2
            center_y = height/2
            # if image is to small to cut from the center, image is cut by the image corner
            if center_x < outputWidth or center_y < outputHeight:
                left = width - outputWidth
                top = 0
                right = width
                bottom = outputHeight
            else:
                left = center_x
                top = center_y - outputHeight
                right = center_x + outputWidth
                bottom = center_y

            im = im.crop((left, top, right, bottom))
            im.save(pathCroppedVal + item[:-4] + '_3.png', 'PNG', quality=100)
    return

def crop_rightBottom(outputWidth = 512, outputHeight = 512, pngs=[], pngsVal=[]):
    for item in pngs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            # crop left top of the image
            width, height = im.size
            center_x = width/2
            center_y = height/2
            # if image is to small to cut from the center, image is cut by the image corner
            if center_x < outputWidth or center_y < outputHeight:
                left = width - outputWidth
                top = height - outputHeight
                right = width
                bottom = height
            else:
                left = center_x 
                top = center_y
                right = center_x + outputWidth
                bottom = center_y + outputHeight

            im = im.crop((left, top, right, bottom))
            im.save(pathCropped + item[:-4] + '_4.png', 'PNG', quality=100)

    for item in pngsVal:
        if os.path.isfile(pathVal + item):
            im = Image.open(pathVal + item)
            # crop left top of the image
            width, height = im.size
            center_x = width/2
            center_y = height/2
            # if image is to small to cut from the center, image is cut by the image corner
            if center_x < outputWidth or center_y < outputHeight:
                left = width - outputWidth
                top = height - outputHeight
                right = width
                bottom = height
            else:
                left = center_x 
                top = center_y
                right = center_x + outputWidth
                bottom = center_y + outputHeight

            im = im.crop((left, top, right, bottom))
            im.save(pathCroppedVal + item[:-4] + '_4.png', 'PNG', quality=100)
    return

def crop(outputWidth = 512, outputHeight = 512, pieces = 1):
    global pathCroppedVal
    global pathCropped
    pieces = int(pieces)
    #select directory
    if pieces == 1:
        pathCropped = pathCropped + "Resized/"
        pathCroppedVal = pathCroppedVal + "Resized/"
    else:
        pathCropped = pathCropped + "ResizedInPieces/"
        pathCroppedVal = pathCroppedVal + "ResizedInPieces/"

    pngs = []
    for file in os.listdir(path=path):
        if file.endswith(".png"):
            pngs.append(file)
    pngsVal = []
    for file in os.listdir(path=pathVal):
        if file.endswith(".png"):
            pngsVal.append(file)

    crop_center(outputWidth, outputHeight, pngs, pngsVal)  

    if pieces > 1 and pieces < 6:
        crop_leftTop(outputWidth, outputHeight, pngs, pngsVal)
        if pieces > 2: crop_rightTop(outputWidth, outputHeight, pngs, pngsVal)
        if pieces > 3: crop_leftBottom(outputWidth, outputHeight, pngs, pngsVal)
        if pieces > 4: crop_rightBottom(outputWidth, outputHeight, pngs, pngsVal)

    print("finish")
    return
    

if __name__ == '__main__':
    crop(512, 512, sys.argv[1])