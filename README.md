# UNI-AIS-MediaDataFormats

II Compress Test Data to a fixed filesize and use WebP, HEIC/HEIF, JPEG XR

### All Section:
#### JPEG:
Generally uses **Discrete Cosine Transformation** and **Huffmann-code** to
compress images (Done in 8x8 blocks).  
Software and description gets provided by the [JPEG org](https://jpeg.org/jpeg/index.html).

#### JPEG2000:
Instead of dividing an image into 8x8 blocks and compressing them individually, JPEG2000 uses a
wavelet transform to divide an image into smaller subbands, each representing a different frequency range.
JPEG2000 also includes several advanced features, including scalability.
Software and description gets provided by the [JPEG org](https://jpeg.org/jpeg2000/index.html).

#### JPEG XL:
Developed by the JPEG Committee, combination of **Google Pik Image** and **FUIF**. 
JPEG XL is a modern image compression standard that uses a modular approach, advanced color science
and new compression techniques to provide better compression efficiency, higher quality images
and more advanced features than existing standards like JPEG. It supports progressive decoding,
animation, and a wide range of color spaces, and is an open-source standard that is continuously improving.
Software and description gets provided by the [JPEG org](https://jpeg.org/jpegxl/index.html).

---
### Project (AI) Section:
#### WebP:
Image format for Web, developed by Google Information ans source code can be 
downloaded [here](https://developers.google.com/speed/webp).  

#### HEIC/HEIF:
**High Efficiency Image Format** which uses the **High Efficiency Image Codec** launched by Apple
with iOS 11. Source code can be found [here](https://github.com/strukturag/libheif)

#### JPEG XR:
Software and description gets provided by the [JPEG org](https://jpeg.org/jpegxr/index.html).