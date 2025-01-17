import os
import re
import random


class FileWalker:
    """
    input:
    dir: Path of the folder
    filetype: '.png' or '.tif' or '.txt' any other type; can enter multiple types at the same time, '.tif','.png'
    fileexception: 'exception' or any other file/folder in which you don't want to consider

    Returns an object with following variables:-
    index: 0,1,2...
    roots: full path of the image
    """

    def __init__(self, dir, *filetype, folderexception = 'exception',fileexception=['fileexception'], print_filecount=True,shuffle = False):

        
        self.id = []  # a1c1d8c2-9915-11ee-9103-bbb8eae05561
        self.roots = []
        self.names = [] # rgb.jpg/ mask.npy etc
        self.namewithoutfiletype = [] # rgb/ mask
        self.index = []  # 0,1,2...

        idx = 0
        for mainroot, maindirs, mainfiles in os.walk(dir):
            for name in mainfiles:
                if folderexception in mainroot:
                    continue
                else:
                    if name in fileexception:
                        continue
                    if name.endswith(filetype):
                        namewithoutfiletype = name.split('.')[0]
                        
                        individualroot = os.path.join(mainroot, name)
                        
                        self.id.append(os.path.split(mainroot)[-1])
                        self.roots.append(individualroot)
                        self.names.append(name)
                        self.namewithoutfiletype.append(namewithoutfiletype)
                        self.index.append(idx)

                        idx = idx+1
                    else:
                        pass
        if print_filecount == True:
            print(
                f"Number of {filetype} files in {os.path.split(dir)[-1]}: {len(self.roots)}")
        if shuffle == True:
            # TODO
            shuffledlists = list(zip(self.id,
                                     self.roots,
                                     self.names,
                                     self.namewithoutfiletype,
                                     self.index))
            random.shuffle(shuffledlists)
            self.id,self.roots,self.names,self.namewithoutfiletype,self.index = zip(*shuffledlists)
        if len(self.roots) == 0:
            raise Exception(f"Number of {filetype} files in {dir}: {len(self.roots)}")
    def __repr__(self) -> str:
        info = """ A class with the following attributes:
name: 
namewithoutfiletype
id: 
index: 0,1,2...
roots: full path of the image
"""
        return info