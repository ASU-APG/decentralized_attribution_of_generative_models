import os
import torch
import torch.nn as nn
from pathlib import Path

#This file is crucial for versatility.
#This file controls different GANs.

class MyUtils():

    def __init__(self, args):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.gan_type = args.GAN_type
        self.project_path = os.getcwd() + '/'
        self.experiment = args.experiment
        self.b_size = args.batch_size
        self.model = None  #Model object only for PGAN
        self.avail_GAN_list = ['PGAN', 'CycleGAN', 'DCGAN']
        self.dataset = args.dataset
        self.is_theory = args.is_theory


    def generator_getter(self, args):
        if (self.gan_type == 'PGAN'):
            from my_utils.my_PGAN_utils import load_pretrained_PGAN

            # Load GAN model
            model, generator = load_pretrained_PGAN(args.dataset, self.project_path)
            self.model = model
            generator = model.getOriginalG()
            generator = nn.DataParallel(generator)

            self.image_size = self.model.getSize()[0]

        elif(self.gan_type == 'CycleGAN'):
            from data import create_dataset
            from models import create_model
            # Load Generator
            model = create_model(args)  # create a model given opt.model and other options
            model.setup(args)  # regular setup: load and print networks; create schedulers
            generator = model.get_generator()

            if(len(args.gpu_ids) > 1):
                generator = nn.DataParallel(generator)

            self.image_size = args.crop_size #image size for CycleGAN

        elif(self.gan_type == 'DCGAN'):
            nz = 100
            ngf = 64

            if(not self.is_theory):
                from my_models.DCGAN import Generator

                if(self.dataset == "MNIST"):
                    nc = 1
                else:
                    nc = 3

                generator = Generator(1, nz, ngf, nc)  # Number of GPU 1 is enough
                gan_path = os.getcwd() + "/weight"
                generator = self.load_weight(generator, gan_path)

                self.image_size = 64

            elif(self.is_theory):
                from my_models.DCGAN_theory import Generator
                if(self.dataset == "MNIST"):
                    nc = 1
                else:
                    nc = 3
                generator = Generator(1, nz, ngf, nc)  # Number of GPU 1 is enough
                gan_path = os.getcwd() + "/weight"
                generator = self.load_weight(generator, gan_path)
                self.image_size = 32
        elif(self.gan_type == "DCGAN_128"):
            nz = 100
            ngf = 64

            from my_models.DCGAN_128_resoulution import Generator

            nc = 3

            generator = Generator(1, nz, ngf, nc)  # Number of GPU 1 is enough
            gan_path = os.getcwd() + "/weight"
            generator = self.load_weight(generator, gan_path)

            self.image_size = 128

        else:
            raise ValueError("Not avail GAN model")

        return generator.to(self.device)


    def load_weight(self, generator, gan_path):
        generator_weight_path = gan_path
        state_dict_mine = torch.load(generator_weight_path + '/generator.pth', map_location='cuda')

        if(self.gan_type == "PGAN"):
            import models.utils.utils as utils  # This lib is from original PGAN implementation.
            try:
                utils.loadPartOfStateDict(generator, state_dict_mine)
            except:
                netG_lp_2 = nn.DataParallel(generator)
                utils.loadPartOfStateDict(netG_lp_2, state_dict_mine)
        elif(self.gan_type == "CycleGAN"):
            try:
                generator.load_state_dict(state_dict_mine)
            except:
                generator = nn.DataParallel(generator)
                generator.load_state_dict(state_dict_mine)

        elif(self.gan_type == "DCGAN" or self.gan_type == "DCGAN_128"):
            try:
                generator.load_state_dict(state_dict_mine)
            except:
                generator = nn.DataParallel(generator)
                generator.load_state_dict(state_dict_mine)


        return generator.to(self.device)

    def model_freezer(self, generator):

        for param in generator.parameters():
            param.requires_grad = False
        generator.eval()

        return generator

    def model_de_freezer(self, generator):
        generator.train()
        for param in generator.parameters():
            param.requires_grad = True

        return generator

    def noise_maker(self ,b_size = None):

        if(b_size is not None):
            batch_size = b_size
        else:
            batch_size = self.b_size

        if(self.gan_type == 'PGAN'):
            noise, _ = self.model.buildNoiseData(batch_size)
        elif(self.gan_type == 'DCGAN' or self.gan_type == 'DCGAN_128'):
            noise =  torch.randn(batch_size, 100, 1,1)
        else:
            raise ValueError("Not avail GAN model")

        return noise.to(self.device)

    def get_image_size(self):
        if(self.gan_type in self.avail_GAN_list):
            return self.image_size
        else:
            raise ValueError("Not avail GAN model")

    def get_data_loader(self, args):
        if(self.gan_type == "PGAN" or self.gan_type == "DCGAN" or self.gan_type == "DCGAN_128"):
            from data_prepare.data_prepare import Data_preparation
            home_path = str(Path.home())
            data_prep = Data_preparation()
            dataloader = data_prep.get_dataloader(args, home_path)

        elif(self.gan_type == "CycleGAN"):
            from data import create_dataset
            dataloader = create_dataset(args)  # create a dataset given opt.dataset_mode and other options
            dataset_size = len(dataloader)  # get the number of images in the dataset.
            print('The number of training images = %d' % dataset_size)

        else:
            raise ValueError("Not avail GAN model")

        return dataloader

    def load_trained_generator(self, netG):

        generator_weight_path = self.project_path + self.experiment
        state_dict_mine = torch.load(generator_weight_path + '/generator.pth', map_location='cuda')

        if(self.gan_type == "PGAN"):
            import models.utils.utils as utils
            try:
                utils.loadPartOfStateDict(netG, state_dict_mine)
            except:
                netG = nn.DataParallel(netG)
                utils.loadPartOfStateDict(netG, state_dict_mine)
        elif(self.gan_type == "CycleGAN"):
            try:
                netG.load_state_dict(state_dict_mine)
            except:
                netG = nn.DataParallel(netG)
                netG.load_state_dict(state_dict_mine)
        else:
            raise ValueError("Not Avail GAN model")

        return netG

    def get_key_index_from_file(self, folder):
        key_index = []
        key = ''
        for i in folder:
            if (i == '_'):
                break
            key_index.append(i)

        for i in key_index[1:]:
            key += i

        return key

    def get_keys(self, how_many_folders, from_path,suffix, device):
        keys = []
        for i in range(how_many_folders):
            folder = 'g' + str(i + 1) + '_k' + str(i + 1)
            keys.append(torch.load(from_path + folder + suffix + 'key_{0}.pth'.format(i + 1), map_location=device))
        for i in range(how_many_folders):
            keys[i] = keys[i].squeeze()

        return keys