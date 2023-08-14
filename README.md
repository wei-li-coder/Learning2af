# Learning2af
Folder 'preprocess_data' contains codes for preprocessing the public dataset and making my own dataset for train.

Folder 'train' contains codes for training with MobileNet_V2 in this dataset.

Folder 'visualization' contains visualizations for patches in a focal stack. In these visualizations, the left patch represents the green channel, the middle patch represents the corresponding in-focus patch, and the right patch represents the filtered-out patch. It is important to note that the left and middle patches are in the same position within each scene. So does the right patch.

progress.csv contains the whole training progress.

__Update in 08.13__ Now it seems the model still had a poor performance on the testing set.
__Update in 08.14__ After running the model for more epochs, it seems that the previous problem has resurfaced. This is puzzling as the preprocessing codes for the training and testing sets remain identical, with the only difference being the data directory. The model's performance, however, appears to differ significantly between the two sets.

