# Batch-Spectral-Penalization

## Prerequisites:

* Python3
* PyTorch == 0.4.0/0.4.1 (with suitable CUDA and CuDNN version)
* torchvision >= 0.2.1
* Numpy
* argparse
* PIL

## Dataset:

You need to modify the path of the image in every ".txt" in "./data".

## Training on one dataset:

All the parameters are set as the same as parameters mentioned in the article. 
You can use the following commands to the tasks:

python -u train.py --gpu_id n --src src --tgt tgt

n is the gpu id you use, src and tgt can be chosen as in "dataset_list.txt".

## Citation:

If you use this code for your research, please consider citing:

```
@inproceedings{BSP_ICML_19,
  title={Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation},  
  author={Chen, Xinyang and Wang, Sinan and Long, Mingsheng and Wang, Jianmin}, 
  booktitle={International Conference on Machine Learning}, 
  pages={1081--1090}, 
  year={2019} 
}
```
## Contact
If you have any problem about our code, feel free to contact chenxinyang95@gmail.com.
