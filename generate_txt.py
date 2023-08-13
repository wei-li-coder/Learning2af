import os
import random

train_dir = '/data/wl/autofocus/learn2focus/dataset/train_demo_new_1/'
dirs_1 = os.listdir(train_dir)
dirs_1.sort(key=int) # 按照字符串数值排序
# dirs_1 = ['0', '1', ..., '48']
train_txt = open('/data/wl/autofocus/learn2focus/dataset/train_demo_new_1.txt','a')
for dir_1 in dirs_1:
    files = os.listdir(train_dir + dir_1)
    files.sort()
    for file in files:
         if random.random() < 0.125:
            name =  train_dir +  dir_1 + '/' + file + ' ' + dir_1 +'\n'
            train_txt.write(name)

train_txt.close()

test_dir = '/data/wl/autofocus/learn2focus/dataset/test_demo_new_1/'
dirs_1 = os.listdir(test_dir)
dirs_1.sort(key=int) # 按照字符串数值排序
# dirs_1 = ['0', '1', ..., '48']
test_txt = open('/data/wl/autofocus/learn2focus/dataset/test_demo_new_1.txt','a')
for dir_1 in dirs_1:
    files = os.listdir(test_dir + dir_1)
    files.sort()
    for file in files:
        # test when testing set is training set
        if random.random() < 0.125:
            name =  test_dir +  dir_1 + '/' + file + ' ' + dir_1 +'\n'
            test_txt.write(name)

test_txt.close()


# label = 0
# while(label < 49):#1024为我们的类别数
#     dir = './data1/images/'#图片文件的地址
#     #os.listdir的结果就是一个list集，可以使用list的sort方法来排序。如果文件名中有数字，就用数字的排序
#     files = os.listdir(dir)#列出dirname下的目录和文件
#     files.sort()#排序
#     train = open('./data1/train.txt','a')
#     text = open('./data1/text.txt', 'a')
#     i = 1
#     for file in files:
#         if i<200000:
#             fileType = os.path.split(file)#os.path.split()：按照路径将文件名和路径分割开
#             if fileType[1] == '.txt':
#                 continue
#             name =  str(dir) +  file + ' ' + str(int(label)) +'\n'
#             train.write(name)
#             i = i+1
           
#         else:
#             fileType = os.path.split(file)
#             if fileType[1] == '.txt':
#                 continue
#             name = str(dir) +file + ' ' + str(int(label)) +'\n'
#             text.write(name)
#             i = i+1
#     text.close()
#     train.close()
