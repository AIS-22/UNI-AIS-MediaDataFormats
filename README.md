# UNI-AIS-MediaDataFormats

Repositiory for the Media Data Formats Pro Seminar Group with the topic Detection of Algorith used in Image Compression with the following Codecs:

- JPEG XL
- WebP
- HEIC/HEIF
- JPEG XR (with overlap Parameter 0, 1 & 2)
- BPG
- AVIF
- Additional Codecs added later
  - JPEG
  - JPEG 2000

## Description

The Goal of this project was to first compress images to a specified Filesize. This means that an image would be compressed to a file size of for example 5 KB by dialing in the quality setting of the encoder to get just below this threshold. Due to the fact that every image is different one need to figure out the matching quality setting for each image sepereatly. To reduce the calcualtion time a divide and conquer approach was used.

In short the image was saved with the quality setting at half of the possible quality values. Then the resulting filesize is determied if its larger than determined the quality will be reduced to the lower half else to the upper half. E.g. First run with Quality 50 -> File size to large next quality step will be 25. If the file size is under the threshhold the next Quality step would be hafway between 25 and 50. And so on until the desired file size is reached for a quality setting with 100 values this would lead to a result within 8 steps.

Since every compression algorithm has different charachterristics at different leves of compression there is a need to compress to differnt file sizes. The ituition beeing that a more compressed image would be easier to classify than a less compressed since the artifacts inherent to the algorithm would not be as strong. The files sizes used were: 5, 10, 17, 25, 32, (40, 50, 60, 75, 100). while the Former ones were used to train the models and the ladder ones only to observe the accuracy of the network with less compressed images.

To classify the compressed images the compressed images were converted back into a png so all the compression types would behave the same. To classify the compressed images a resnet-18 in pyytorch was used. On the one hand with transfere learning and on the other hand by starting from scrach.

## Software & Preprocessing

### Used Software

- General Requirements

    ```sh
    # For Preprocessing
    conda install pillow
    pip install imagecodecs # min version 2023.3.16
    # Training
    conda install torch torchvision
    ```

- Pillow
  - JPEG
  - WebP
- Pillow Plugins
  - AVIF [Pip Pillow AVIV Plugin](https://pypi.org/project/pillow-avif-plugin/)

    ```sh
    pip install pillow-avif-plugin
    ```

  - HEIC [Pip Pillow HEIF Plugin](https://pypi.org/project/pillow-heif/)

    ```sh
    pip install pillow-heif
    ```

- apt
  - JPEG 2000

    ```sh
    sudo apt install libopenjp2-7 libopenjp2-tools
    ```

  - JPEG XR

    ```sh
    sudo apt-get install libjxr-dev
    sudo apt-get install libjxr-tools
    ```

- Homebrew / Linux Brew for
  - BPG

    ```sh
    brew install libbpg
    ```

  - JPEX XL

    ```sh
    brew install jpeg-xl
    ```

### Running the Scripts

To runn the preprocessing the `preprocessingAll.py` script has to be executed. This will first crop the images t the 512x512 pixel size on witch the file sizes were based. After the crop the images ware first compressed to the right file size and afterwards saved as a png in the decoded folder.

To train the network use eatither the `cnnTrain.py`, `pretrainedCNN.py` or the `pretrainedCNNFilesize.py` scripts. These will either start the training form 0, use the pretrained resnet 18 weights for all files sizes or just a single file size. The pretrained networks will save the model weights to the models folder.

## Results

### PSNR over Compression Ratio of the Codecs (Mean over 5 images)

![PSNR](Plots/psnr.png)

## Authors

- Stefan Findenig
- Michael Hafner
- Moritz Langer
- Aleksander Radovic
