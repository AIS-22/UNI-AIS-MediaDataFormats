import asyncio
import avifenc
import bpgenc
import cropImages
import heicEnc
import jxrenc_0
import jxrenc_1
import jxrenc_2
import webP
import jpegxl
import jpeg2000enc
import jpegenc
from concurrent.futures import ThreadPoolExecutor


async def preprocess(cropNeeded=False,
                     width=512,
                     height=512,
                     maxFileSizeKb=5,
                     printProgress=True):
    if cropNeeded:
        # first crop images
        print('Start to crop images to {}x{} pixel.'.format(width, height))
        cropImages.crop(width, height)

    filesizes = [5, 10, 17, 25, 32, 40, 50, 60, 75, 100]
    for maxFileSizeKb in filesizes:
        # start the encoding and decoding task of each codec asynchronously
        print('Start to encode images. Filesize='+str(maxFileSizeKb))
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(avifenc.encode_avif, printProgress, maxFileSizeKb),
                executor.submit(bpgenc.encode_bpg, printProgress, maxFileSizeKb),
                executor.submit(heicEnc.encode_heic, printProgress, maxFileSizeKb),
                executor.submit(jpegenc.encode_jpeg, printProgress, maxFileSizeKb),
                executor.submit(jpeg2000enc.encode_jpeg2k, printProgress, maxFileSizeKb),
                executor.submit(jpegxl.encode_jpgxl, printProgress, maxFileSizeKb),
                executor.submit(jxrenc_0.encode_jxr, printProgress, maxFileSizeKb),
                executor.submit(webP.encode_webp, printProgress, maxFileSizeKb),
            ]
            for future in futures:
                future.result()
        jxrenc_1.encode_jxr(printProgress, maxFileSizeKb)
        jxrenc_2.encode_jxr(printProgress, maxFileSizeKb)
        print('Finished preprocessing job.')


if __name__ == '__main__':
    asyncio.run(preprocess())
