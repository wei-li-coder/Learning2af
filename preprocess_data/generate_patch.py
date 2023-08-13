import numpy as np
import cv2
from patchify import patchify
import bisect

def takeClosest(myList, myNumber):
    # choose nearest distance to decide index for the patch
    if (myNumber >= myList[-1]):
        return myList[-1]
    elif myNumber <= myList[0]:
        return myList[0]
    pos = bisect.bisect_left(myList, myNumber)
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return 48-pos
    else:
       return 48-(pos-1)

def cal_groundtruth_index(patch, focal_distance):
    patch = patch/255
    # in mm
    max = 100000.0
    min = 100.0

    depth = (max * min) / (max - (max - min) * patch)
    index = takeClosest(focal_distance, np.median(depth))
    return index
