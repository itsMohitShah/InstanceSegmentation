import tarfile

with tarfile.open('dl_challenge.tar.xz') as f:
    f.extractall()