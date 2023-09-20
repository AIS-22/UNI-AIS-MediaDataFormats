#!/bin/bash

cd Images/DIV2K_train_HR/Decoded

filesize=$1

cp $filesize/AVIF/*.png  all/AVIF
cp $filesize/BPG/*.png  all/BPG
cp $filesize/HEIC/*.png  all/HEIC
cp $filesize/JPEG/*.png  all/JPEG
cp $filesize/JPEG2000/*.png  all/JPEG2000
cp $filesize/JPEG_XL/*.png  all/JPEG_XL
cp $filesize/JPEG_XR_0/*.png  all/JPEG_XR_0
cp $filesize/JPEG_XR_1/*.png  all/JPEG_XR_1
cp $filesize/JPEG_XR_2/*.png  all/JPEG_XR_2
cp $filesize/WEBP/*.png  all/WEBP

cd ../../..
cd Images/DIV2K_valid_HR/Decoded

cp $filesize/AVIF/*.png  all/AVIF
cp $filesize/BPG/*.png  all/BPG
cp $filesize/HEIC/*.png  all/HEIC
cp $filesize/JPEG/*.png  all/JPEG
cp $filesize/JPEG2000/*.png  all/JPEG2000
cp $filesize/JPEG_XL/*.png  all/JPEG_XL
cp $filesize/JPEG_XR_0/*.png  all/JPEG_XR_0
cp $filesize/JPEG_XR_1/*.png  all/JPEG_XR_1
cp $filesize/JPEG_XR_2/*.png  all/JPEG_XR_2
cp $filesize/WEBP/*.png  all/WEBP