#!/bin/bash
codecs=( AVIF BPG HEIC JPEG JPEG2000 JPEG_XL JPEG_XR_0 JPEG_1 JPEG_XR_1 JPEG_XR_2 WEBP )
echo "train"
echo "param"
cd Images/DIV2K_train_HR
# find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
#   printf "%-25.25s : " "$dir"
#   find "$dir" -type f | wc -l
# done

echo "decoded"
cd Decoded_pieces
find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
  printf "%-25.25s : " "$dir"
  find "$dir" -type f | wc -l
done

echo $1
cd $1
find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
  printf "%-25.25s : " "$dir"
  find "$dir" -type f | wc -l
done

echo "valid"
echo "param"
cd ../../../..
cd Images/DIV2K_valid_HR
# find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
#   printf "%-25.25s : " "$dir"
#   find "$dir" -type f | wc -l
# done

echo "decoded"
cd Decoded_pieces
find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
  printf "%-25.25s : " "$dir"
  find "$dir" -type f | wc -l
done

echo $1
cd $1
find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
  printf "%-25.25s : " "$dir"
  find "$dir" -type f | wc -l
done