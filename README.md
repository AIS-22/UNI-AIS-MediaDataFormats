# UNI-AIS-MediaDataFormats

II Compress Test Data to a fixed filesize and use JPEG XL, WebP, HEIC/HEIF, JPEG XR, BPG, AVIF

Afterwards Classify the different Compression Algorithms by an ML Model.

## Compression Algorithms

### JPEG (handeled by other group)

Generally uses **Discrete Cosine Transformation** and **Huffmann-code** to
compress images (Done in 8x8 blocks).  
Software and description gets provided by the [JPEG org](https://jpeg.org/jpeg/index.html).

### JPEG2000 (handeled by other group)

Instead of dividing an image into 8x8 blocks and compressing them individually, JPEG2000 uses a
wavelet transform to divide an image into smaller subbands, each representing a different frequency range.
JPEG2000 also includes several advanced features, including scalability.
Software and description gets provided by the [JPEG org](https://jpeg.org/jpeg2000/index.html).

### JPEG XL

Developed by the JPEG Committee, combination of **Google Pik Image** and **FUIF**. 
JPEG XL is a modern image compression standard that uses a modular approach, advanced color science
and new compression techniques to provide better compression efficiency, higher quality images
and more advanced features than existing standards like JPEG. It supports progressive decoding,
animation, and a wide range of color spaces, and is an open-source standard that is continuously improving.
Software and description gets provided by the [JPEG org](https://jpeg.org/jpegxl/index.html).

### WebP

Image format for Web, developed by Google Information ans source code can be 
downloaded [here](https://developers.google.com/speed/webp).  

### HEIC/HEIF

**High Efficiency Image Format** which uses the **High Efficiency Image Codec** launched by Apple
with iOS 11. Source code can be found [here](https://github.com/strukturag/libheif)

Pillow-heif is the implementation of heic on top of PIL

### JPEG XR

Software and description gets provided by the [JPEG org](https://jpeg.org/jpegxr/index.html).

### BPG

Needs to be installed locally using the github repo described in [#13](https://github.com/AIS-22/UNI-AIS-MediaDataFormats/issues/13)

### AVIF

PIL with avif can be used here, therefore just install pillow-avif-plugin python lib and import pillow_avif at the very beginning. Read description of [#7](https://github.com/AIS-22/UNI-AIS-MediaDataFormats/issues/7)
