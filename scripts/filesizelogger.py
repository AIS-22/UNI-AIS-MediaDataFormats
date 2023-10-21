import os

def log_filesize(filesize, target, codec, log_file='filesize_log.txt'):
    with open('filesize_log.txt', 'a') as f:
        f.write(f'{filesize}, {target}, {codec}\n')