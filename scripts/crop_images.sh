#!/bin/bash

resizedPath=../Images/Resized/
filePathTrain=../Images/DIV2K_train_HR/
resizedTrainPath=../Images/Resized/DIV2K_train_HR/
W=512
H=512

if [ ! -e $resizedPath ]; then
  mkdir $resizedPath
fi

if [ ! -e $resizedTrainPath ]; then
  mkdir $resizedTrainPath
### only delete if files exists in path
elif [ "$(ls $resizedTrainPath | wc -l)" != 0 ]; then
  echo "Deleting all existing files in resized train path"
  rm $resizedTrainPath*.png
fi

echo "Resizing training images..."
### crop the files in train path
for file in $(ls $filePathTrain); do
  newfile=$resizedTrainPath$file
  convert $filePathTrain$file -crop "$W"x"$H"+0+0 $newfile
done

filePathVal=../Images/DIV2K_valid_HR/
resizedValPath=../Images/Resized/DIV2K_valid_HR/

if [ ! -e $resizedValPath ]; then
  mkdir $resizedValPath
### only delete if files exists in path
elif [ "$(ls $resizedValPath | wc -l)" != 0 ]; then
  echo "Deleting all existing files in resized val path"
  rm $resizedValPath*.png
fi

echo "Resizing validation images..."
### crop the files in val path
for file in $(ls $filePathVal); do
  newfile=$resizedValPath$file
  convert $filePathVal"$file" -crop "$W"x"$H"+0+0 $newfile
done

exit 0
