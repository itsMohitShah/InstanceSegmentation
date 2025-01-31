import tarfile
import os
destination = r'DL_Challenge'
os.makedirs(destination, exist_ok=True)
with tarfile.open(r'.\dl_challenge.tar.xz') as f:
    f.extractall(destination)
print("Execution complete")