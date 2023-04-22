import asyncio
import avifenc
import bpgenc
import cropImages
import heicEnc
import jxrenc
import webP
from concurrent.futures import ThreadPoolExecutor

maxFileSizeKb = 32
printProgress = True
WIDTH = 512
HEIGHT = 512


async def preprocess():
    # first crop images
    print('Start to crop images.')
    cropImages.crop(WIDTH, HEIGHT)

    # start the encoding and decoding task of each codec asynchronously
    print('Start to encode images.')
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(avifenc.encode_avif, printProgress, maxFileSizeKb),
            executor.submit(bpgenc.encode_bpg, printProgress, maxFileSizeKb),
            executor.submit(heicEnc.encode_heic, printProgress, maxFileSizeKb),
            executor.submit(jxrenc.encode_jxr, printProgress, maxFileSizeKb),
            executor.submit(webP.encode_webp, printProgress, maxFileSizeKb),
        ]
        for future in futures:
            future.result()
    print('Finished preprocessing job.')


if __name__ == '__main__':
    asyncio.run(preprocess())
