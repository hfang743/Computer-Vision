import numpy as np
from os import walk
from __main__ import *

from os import listdir
from os.path import isfile, join

# imagelist=np.array(['0000000000'])


# f = []
# for (dirpath, dirnames, filenames) in walk(mypath):
#     f.extend(filenames)
#
#     print(f)
#     break

def imagefiles(mypath):
    # (_, _, filenames) = walk(directory).next()
    imageList = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    return imageList
