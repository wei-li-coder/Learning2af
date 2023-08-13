import os
import random

# generate txt
train_dir = '/data/wl/autofocus/learn2focus/dataset/train_set/'
dirs_1 = os.listdir(train_dir)
dirs_1.sort(key=int)
# dirs_1 = ['0', '1', ..., '48']
train_txt = open('/data/wl/autofocus/learn2focus/dataset/train_set.txt','a')
for dir_1 in dirs_1:
    files = os.listdir(train_dir + dir_1)
    files.sort()
    for file in files:
         if random.random() < 0.125:
            name =  train_dir +  dir_1 + '/' + file + ' ' + dir_1 +'\n'
            train_txt.write(name)

train_txt.close()

test_dir = '/data/wl/autofocus/learn2focus/dataset/test_set/'
dirs_1 = os.listdir(test_dir)
dirs_1.sort(key=int)
# dirs_1 = ['0', '1', ..., '48']
test_txt = open('/data/wl/autofocus/learn2focus/dataset/test_set.txt','a')
for dir_1 in dirs_1:
    files = os.listdir(test_dir + dir_1)
    files.sort()
    for file in files:
        name =  test_dir +  dir_1 + '/' + file + ' ' + dir_1 +'\n'
        test_txt.write(name)

test_txt.close()
