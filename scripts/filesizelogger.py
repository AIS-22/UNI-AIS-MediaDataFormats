import os

def log_filesize(file, log_file='filesize_log.txt'):
    filesize = os.path.getsize(file)
    with open('filesize_log.txt', 'a') as f:
        f.write(file + ' ' + str(filesize) + '\n')