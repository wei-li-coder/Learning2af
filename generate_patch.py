import numpy as np
import cv2
from patchify import patchify
import bisect

def takeClosest(myList, myNumber):
    if (myNumber >= myList[-1]):
        return myList[-1]
    elif myNumber <= myList[0]:
        return myList[0]
    pos = bisect.bisect_left(myList, myNumber)   # 找到 mylist 里面第一个不比 mynumber 小（即 >= ）的数的索引下标
    # 返回的插入点 pos 可以将数组myList分成两部分。左侧是 all(val < x for val in myList[lo:i]) ，右侧是 all(val >= x for val in myList[i:hi])
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       # slice index是倒序排列的
       # 选择after的位置
       return 48-pos
    else:
       # 选择before的位置
       return 48-(pos-1)

def get_idx_from_depth(focal_distance, pdepth):
    pdiopters = 1.0 / pdepth
    focal_diopters = np.array([1.0 / d for d in focal_distance])
    return int(round(np.interp(pdiopters, focal_diopters, np.arange(focal_diopters.shape[0]))))


def cal_groundtruth_index(patch, focal_distance):
    patch = patch/255
    max = 100000.0
    min = 100.0
    # max = 100
    # min = 0.1

    depth = (max * min) / (max - (max - min) * patch)

    index = takeClosest(focal_distance, np.median(depth))
    
    return index



# focal_distance = [102.01, 104.23, 106.54, 108.47, 110.99, 113.63, 116.40, 118.73, 121.77,\
#                  124.99, 127.69, 131.23, 134.99, 138.98, 142.35, 146.81, 150.59, 155.61,\
#                  160.99, 165.57, 171.69, 178.29, 183.96, 191.60, 198.18, 207.10, 216.88,\
#                  225.41, 237.08, 247.35, 261.53, 274.13, 291.72, 307.54, 329.95, 350.41,\
#                  379.91, 407.40, 447.99, 486.87, 546.23, 605.39, 700.37, 801.09, 935.91,\
#                  1185.83, 1508.71, 2289.27, 3910.92]

# rawimg_dep = cv2.imread('/data/wl/autofocus/learn2focus/dataset/train/train1/merged_depth/apt1_0/result_merged_depth_center.png', cv2.IMREAD_UNCHANGED)
# patches_dep = patchify(rawimg_dep, (128,128), step=40)

# idx = cal_groundtruth_index(patches_dep[5,4,:,:], focal_distance)
# print(idx)