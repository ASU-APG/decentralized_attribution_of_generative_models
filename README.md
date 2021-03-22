## Decentralized Attribution of Generative Models
[Paper](https://arxiv.org/abs/2010.13974)\
Pytorch implementation for "Decentralized Attribution of Generative Models" (ICLR 2021).

<img src = "https://user-images.githubusercontent.com/46791426/111937927-316c9480-8b0c-11eb-92e0-3b1c85e04dfb.png" width="460" height="300">

We develop sufficient conditions for model attribution: Perturbing the authentic dataset in different directions with angles larger than a data-dependent threshold guarantees attributability of the perturbed distributions.\
(a) A threshold of 90 degrees suffices for benchmark datasets. \
(b) Smaller angles may not guarantee attributability.


### Getting Started
Linux \
Python 3.6 \
NVIDIA GPU + CUDA CuDNN

We tested on Ubuntu 16.04.\
For PGAN, we recommend 2 or more GPUs.

To install requirements:

```setup
pip install -r requirements.txt
```

### Set-up for Training
Since this project is tested on various GANs, you need to follow the steps in this section.\
We explain how to set up the experiments.

#### Datasets
Please locate dataset (e.g. CelebA and MNIST) under the your_home_path/deep_data/data. (e.g. /home/[user_name]/deep_data/data/'dataset_name').\
For Cityscapes, you don't need to locate it with other datasets.

1. CelebA\
We preprocessed CelebA following [pytorch official release of PGAN](https://github.com/facebookresearch/pytorch_GAN_zoo).\
You can find a way to set up 'celeba cropped' from the 'Quick training' section in the [given link](https://github.com/facebookresearch/pytorch_GAN_zoo).

2. MNIST\
If you don't have MNIST in the location, the code will download this dataset, automatically. 

3. Cityscapes\
We preprocessed Cityscapes based on [official release of CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).


#### Weights and Essential Folders of Original Repository
In this section, we explain how to set the weights and essential folders for the experiments.\
For PGAN and CycleGAN, you can download the weights from the publicly available repository.\
The links are available in each section. \
But for DCGAN, we already attached the weights under the 'weight' folder.


##### Progressive GAN ([pytorch official release](https://github.com/facebookresearch/pytorch_GAN_zoo))
1. You must change this folder name ('supplementary') to PGAN (e.g. ./supplementary to ./PGAN).
2. Copy and paste 'models' folder from the repo and paste it to the current folder.
3. Download the pre-trained 'celeba_cropped' weight from the link. 
4. Change the name of the weight file from celebaCropped_s5_i83000-2b0acc76.pth to generator.pth and put the file in the 'weight' folder.


##### CycleGAN ([official release](https://github.com/junyanz/CycleGAN))
1. You must change the folder name ('supplementary') to CycleGAN (e.g. ./supplementary to ./CycleGAN).
2. Copy and paste the 'data', 'util', 'models' folder from the repo and paste it into the current folder.
3. Download Cityscapes weights (the official CycleGAN repo will introduce how to download).
4. Change weight file name from 'cityscapes_photo2label_pretrained' to 'latest_net_G_A.pth'.
4. Copy weight in 'cityscapes_label2photo_pretrained' to 'cityscapes_photo2label_pretrained' and rename it to 'latest_net_G_B.pth'.
4. In 'models' folder, you need to add this function at the last line of 'cycle_gan_model.py'.
```add
    def get_generator(self):
        return self.netG_A
```


##### DCGAN
1. You must change the folder name ('supplementary') to DCGAN. (e.g. ./supplementary to ./DCGAN)\
2. (a) Move 'generator.pth' file from 'DCGAN/weight/MNIST' to 'DCGAN/weight'.\
   (b) Move 'generator.pth' file from 'DCGAN/weight/CELEBA' to 'DCGAN/weight'.


### Training
In this section, we introduce how to train the keys and generators.\
If you want to change the hyper-parameters, you can change them in 'sh' folder.\
**Note that you must have changed the 'supplementary' folder name to the GAN model name, as we explained in the Set-up section.**\
For example, when you train CycleGAN, the folder name should be CycleGAN. \
As a result, you will get five decentralized generators.\
For more experiments, you can change the arguments in './sh' folder. \
To train the model(s) in the paper, run these commands:

#### PGAN
```
#Train first generator and key
sh ./sh/PGAN/step1_2.sh

#Train keys
sh ./sh/PGAN/new_key_training.sh

#Train generators corresponding to keys
sh ./sh/PGAN/new_generators.sh
```

#### CycleGAN
```
#Train first generator and key
sh ./sh/CycleGAN/step1_2.sh

#Train keys
sh ./sh/CycleGAN/new_key_training.sh

#Train generators corresponding to keys
sh ./sh/CycleGAN/new_generators.sh
```



#### DCGAN
To train MNIST:
```
#Train first generator and key
sh ./sh/DCGAN_MNIST/step1_2.sh

#Train keys
sh ./sh/DCGAN_MNIST/new_key_training.sh

#Train generators corresponding to keys
sh ./sh/DCGAN_MNIST/new_generators.sh
```


To train CelebA:
```
#Train first generator and key
sh ./sh/DCGAN_CELEBA/step1_2.sh

#Train keys
sh ./sh/DCGAN_CELEBA/new_key_training.sh

#Train generators corresponding to keys
sh ./sh/DCGAN_CELEBA/new_generators.sh
```


### Robust training
In this section, we explain how to train the robust generators against post-processes.\
You can run this code for any models:
```Robust Training
sh ./sh/'model_name'/attack.sh
```

During this process, the script will generate folders named after the post-process type (e.g. Blur, Crop).\
To evaluate each of the robust training results, you need to move trained models to the current folder (./[GAN_Name]/).



### Evaluation
In this section, we explain how to use the evaluation script.\
**Even if you just want to evaluate the model, you should locate the generator weight in the weight folder.**\
This is because 'distinguishability' needs the original generator's output images.
```eval
sh ./sh/[model_name]/evaluation.sh
```
The model will give you the result of non-robust distinguishability and attributability.\
You can get a robust result by changing the argument in 'evaluation.sh'.\
Please see the evaluation.sh file in sh folder.


### Pre-trained Models
We uploaded only twenty models for each of GANs.
You can download pre-trained models here:
- [Pre-trained models](https://drive.google.com/drive/folders/1j72a7YtpCM0TJTQz0A4Nr20qPyTj7CHZ?usp=sharing) 


### Related Works
[The reading list](https://github.com/ASU-Active-Perception-Group/awesome_attribution_of_generative_models) introduces related work of attribution of generative models.\
We highly recommend this list to those who want to start to work on attribution of generative models.

### Citation
```bibtex
@inproceedings{
kim2021,
title={ Decentralized Attribution of Generative Models},
author={Changhoon Kim and Yi Ren and Yezhou Yang},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=_kxlwvhOodK}
}
```
