# BSPforDA

## Prerequisites:

* Python3
* PyTorch == 0.4.0/0.4.1 (with suitable CUDA and CuDNN version)
* torchvision >= 0.2.1
* Numpy
* argparse
* PIL

## Dataset:

VisDA 2017 dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public) in the classification track.

You need to modify the path of the image in "train.txt" and "validition.txt".

## Training on VisDA challenge:

All the parameters are set as the same as parameters mentioned in the article. 
You can use the following commands to the tasks:

BSP+DANN: python -u BSP+DANN.py

BSP+CDAN: python -u BSP+CDAN.py
