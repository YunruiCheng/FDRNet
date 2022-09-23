## Installation
The model is built in PyTorch 1.12.1 and tested on Ubuntu 20.04.5 environment (Python3.7, CUDA11.6).

For installing, follow these intructions
```
conda create -n fdrnet python=3.7
conda activate fdrnet
conda install pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3.1 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```
## Training

- Train the model with default arguments by running

```
python train.py
```


## Evaluation

1. Download the [model](https://pan.baidu.com/s/1qqOS5EL_8fvv8BXxmBez5A) (password:a4fj) and place it in `./checkpoints/`

2. Download test datasets (Rain200L, Rain200H, Rain800, Rain1200)  and place them in `./Datasets/`

3. Run
```
python test.py
```

