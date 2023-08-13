import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from patchify import patchify
from patch_my import cal_groundtruth_index
import numpy as np
from random import randint

# focal distance in mm
focal_distance = [102.01, 104.23, 106.54, 108.47, 110.99, 113.63, 116.40, 118.73, 121.77,\
                 124.99, 127.69, 131.23, 134.99, 138.98, 142.35, 146.81, 150.59, 155.61,\
                 160.99, 165.57, 171.69, 178.29, 183.96, 191.60, 198.18, 207.10, 216.88,\
                 225.41, 237.08, 247.35, 261.53, 274.13, 291.72, 307.54, 329.95, 350.41,\
                 379.91, 407.40, 447.99, 486.87, 546.23, 605.39, 700.37, 801.09, 935.91,\
                 1185.83, 1508.71, 2289.27, 3910.92]

# focal_distance.reverse()
# focal_distance = np.array([d / 1000.0 for d in focal_distance])
            
def read_directory(directory_name):
    cnt = 0 
    right_dir = ['result_up_pd_right_bottom.png', 'result_up_pd_right_center.png', 'result_up_pd_right_left.png', 'result_up_pd_right_right.png', 'result_up_pd_right_top.png']
    left_dir = ['result_up_pd_left_bottom.png', 'result_up_pd_left_center.png', 'result_up_pd_left_left.png', 'result_up_pd_left_right.png', 'result_up_pd_left_top.png']
    depth_dir = ['result_merged_depth_bottom.png', 'result_merged_depth_center.png', 'result_merged_depth_left.png', 'result_merged_depth_right.png', 'result_merged_depth_top.png']
    conf_dir = ['result_merged_conf_bottom.exr', 'result_merged_conf_center.exr', 'result_merged_conf_left.exr', 'result_merged_conf_right.exr', 'result_merged_conf_top.exr']
    dirs_1 = os.listdir(directory_name)
    # print(dirs_1)
    dirs_1.sort()
    # dirs_1 = ['train1','train2',...,'train7']
    for dir_1 in dirs_1:
        dir_right_1 = 'raw_up_right_pd/'
        dir_left_1 = 'raw_up_left_pd/'
        dir_depth_1 = 'merged_depth/'
        dir_conf_1 = 'merged_conf/'
        dirs_2 = os.listdir(directory_name + dir_1 + '/' + dir_right_1)
        dirs_2.sort()
        # dirs_2 = ['apt1_0',...] scene name
        for dir_2 in dirs_2:
            dirs_3 = os.listdir(directory_name + dir_1 + '/' + dir_right_1 + dir_2)
            dirs_3.sort(key=int)
            # dirs_3 = ['0','1',...,'48']
            i = []
            j = [] # random sample index
            for _ in range(1):
                i.append(randint(0, 9))
                j.append(randint(0, 6))
            for dir_3 in dirs_3:
                for dir_4 in range(5):
                    rawimg_right = cv2.imread(directory_name +  dir_1 + '/' + dir_right_1 + dir_2 + '/' + dir_3 + '/' + right_dir[dir_4], cv2.IMREAD_UNCHANGED)
                    rawimg_right = cv2.resize(rawimg_right,dsize=None,fx=0.25,fy=0.25,interpolation=cv2.INTER_LINEAR)
                    patches_right = patchify(rawimg_right, (128,128), step=40)
                    rawimg_left = cv2.imread(directory_name +  dir_1 + '/' + dir_left_1 + dir_2 + '/' + dir_3 + '/' + left_dir[dir_4], cv2.IMREAD_UNCHANGED)
                    rawimg_left = cv2.resize(rawimg_left,dsize=None,fx=0.25,fy=0.25,interpolation=cv2.INTER_LINEAR)
                    patches_left = patchify(rawimg_left, (128,128), step=40)
                    rawimg_dep = cv2.imread(directory_name + dir_1 + '/' + dir_depth_1 + dir_2 + '/' + depth_dir[dir_4], cv2.IMREAD_UNCHANGED)
                    patches_dep = patchify(rawimg_dep, (128,128), step=40)
                    rawimg_conf = cv2.imread(directory_name + dir_1 + '/' + dir_conf_1 + dir_2 + '/' + conf_dir[dir_4], cv2.IMREAD_UNCHANGED)
                    patches_conf = patchify(rawimg_conf[:,:,2], (128,128), step=40)
                    for cnt_idx in range(1):
                        if np.median(patches_conf[i[cnt_idx],j[cnt_idx],:,:]) == 1:
                            idx = cal_groundtruth_index(patches_dep[i[cnt_idx],j[cnt_idx],:,:], focal_distance)
                            # save both dp into one image
                            newimg = np.concatenate((patches_left[i[cnt_idx],j[cnt_idx],:,:], patches_right[i[cnt_idx],j[cnt_idx],:,:]), axis = 1)
                            cv2.imwrite('/data/wl/autofocus/learn2focus/dataset/test_demo_new_2' + '/' + str(idx) + '/' + str(cnt) + '_' + dir_3.rjust(2,'0') + '.png', newimg)
                            cnt += 1
                            print(cnt)

########## make directory
def mkd_my():
    path = "/data/wl/autofocus/learn2focus/dataset/test_demo_new_2/" 
    for i in range(0, 49):
        file_dir = path + str(i)
        if os.path.exists(file_dir):
            continue
        else:
            os.makedirs(file_dir)


mkd_my()
read_directory("/data/wl/autofocus/learn2focus/dataset/test/")


