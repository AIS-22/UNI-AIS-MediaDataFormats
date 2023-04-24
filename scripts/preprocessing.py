import asyncio
import avifenc
import bpgenc
import cropImages
import heicEnc
import jxrenc
import webP
import jpegxl
from concurrent.futures import ThreadPoolExecutor


async def preprocess(cropNeeded=True,
                     width=512,
                     height=512,
                     maxFileSizeKb=32,
                     printProgress=True):
    if cropNeeded:
        # first crop images
        print('Start to crop images to {}x{} pixel.'.format(width, height))
        cropImages.crop(width, height)

    # start the encoding and decoding task of each codec asynchronously
    print('Start to encode images.')
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(avifenc.encode_avif, printProgress, maxFileSizeKb),
            executor.submit(bpgenc.encode_bpg, printProgress, maxFileSizeKb),
            executor.submit(heicEnc.encode_heic, printProgress, maxFileSizeKb),
            executor.submit(jxrenc.encode_jxr, printProgress, maxFileSizeKb),
            executor.submit(webP.encode_webp, printProgress, maxFileSizeKb),
            executor.submit(jpegxl.encode_jpgxl, printProgress, maxFileSizeKb),
        ]
        for future in futures:
            future.result()
    print('Finished preprocessing job.')


if __name__ == '__main__':
    asyncio.run(preprocess())
