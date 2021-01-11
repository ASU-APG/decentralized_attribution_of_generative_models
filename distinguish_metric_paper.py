from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
from torch.autograd import Variable
import os
import json
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy
from pathlib import Path

#-------Official Lib----------#

#-------My code Importing------#
#My import
from my_utils.my_utils import MyUtils
from data_prepare.data_prepare import Data_preparation
from attack_methods.attack_initializer import attack_initializer


#------Description---------#


def attacking_images(images, attack):
    if(attack is not None):
        images = attack(images)

    return images


if __name__ == "__main__":
    # Parser
    if (not "CycleGAN" in os.getcwd()):
        from my_options.my_base_option import BaseOptions

        args = BaseOptions().parse()
    elif ("CycleGAN" in os.getcwd()):
        from my_options.CycleGAN.test_options import TestOptions

        args = TestOptions().parse()
        args.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        args.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        args.phase = 'test'
        args.model = 'cycle_gan'
        args.num_threads = 0
        args.batch_size = 100
        number_of_cycle = 1
        args.dataset = None
        args.is_theory = False




    # Distinguishability is not defined with attacking.
    # Attack methods
    attack_methods = ['Blur', 'Crop', 'Jpeg', 'Combination', 'Noise']
    if(args.attack_type in attack_methods):
        attack = attack_initializer(args.attack_type, is_train=False)
    else:
        attack = None
    #attack = None

    # Device Setting
    cudnn.benchmark = True
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # Folder Setting
    project_path = os.getcwd() + '/'
    # os.chdir(project_path)
    suffix = args.experiment[5:]

    # My Utils
    myutils = MyUtils(args)

    # Generator setting
    netG_original = myutils.generator_getter(args)
    netG_original = myutils.model_freezer(netG_original) #Root model




    key_index = np.arange(args.how_many_generator) + 1
    number_of_calculations = 1
    mean = []
    std = []
    distinguishability_list = []

    for i in range(args.how_many_generator):
        #e.g. g1_k1_crop_b0_lp2
        key_path = 'g' + str(i+1) + '_k' + str(i+1)
        #print("------------------current key------------------")
        #print("Key_path: " + key_path + suffix)
        #print("-----------------------------------------------")

        generator_weight_path = project_path + key_path + suffix

        another_key = torch.load(generator_weight_path + '/key_{0}.pth'.format(i+1))
        another_key.to(device)

        netG = myutils.generator_getter(args)
        netG = myutils.load_weight(netG, generator_weight_path)
        netG = myutils.model_freezer(netG)


        for cal in range(number_of_calculations):

            # Data load
            # Data preparation
            dataloader = myutils.get_data_loader(args)

            #Other generator need to be out of distribution
            p_d_mean = 0
            p_g_mean = 0
            sample_size = 1000

            if(args.GAN_type != 'CycleGAN'):

                right_counter = 0
                dataset_size = 0

                for j, data in enumerate(dataloader):
                    if(dataset_size >= sample_size):
                        break
                    real = data[0].to(device)
                    real = attacking_images(real, attack)
                    b_size = real.size(0)
                    real = real.view(b_size, -1)
                    dataset_size = dataset_size + b_size
                    #print(torch.matmul(real, another_key))
                    right_counter = right_counter + torch.sum(torch.matmul(real, another_key) > 0)

                p_d_mean = right_counter.item() / dataset_size
                #print("Distinguishability (P_D): %.5f" % (p_d_mean))
                #print("Data size: " + str(dataset_size))
                #print("number of samples of rightly classified: " + str(right_counter))



                right_counter = 0
                dataset_size = 0

                for j in range(20):
                    with torch.no_grad():
                        b_size = 50
                        noise = myutils.noise_maker(b_size)
                        fake_netG_1 = netG(noise).to(device)
                        fake_netG_1 = attacking_images(fake_netG_1, attack)
                        fake_netG_1 = fake_netG_1.view(b_size, -1)

                        dataset_size = dataset_size + b_size
                        right_counter = right_counter + torch.sum(torch.matmul(fake_netG_1, another_key) < 0)

                p_g_mean = right_counter.item() / dataset_size
                #print("Distinguishability (P_G): %.5f" % (p_g_mean))
                #print("Data size: " + str(dataset_size))
                #print("number of samples of wrongly classified: " + str(right_counter))


            elif(args.GAN_type == 'CycleGAN'):
                right_counter = 0
                dataset_size = 0

                for j, data in enumerate(dataloader):
                    if (j == 1):
                        break
                    real = data['B'].to(device)
                    b_size = real.size()[0]
                    real = real.view(b_size, -1)

                    dataset_size = dataset_size + b_size

                    right_counter = right_counter + torch.sum(torch.matmul(real, another_key) > 0)

                p_d_mean = right_counter.item() / dataset_size
                # print("Distinguishability (P_D): %.5f" % (p_d_mean))
                # print("Data size: " + str(dataset_size))
                # print("number of samples of rightly classified: " + str(right_counter))

                right_counter = 0
                dataset_size = 0

                for j, data in enumerate(dataloader):
                    if(j == 1):
                        break
                    with torch.no_grad():
                        noise = data['A'].to(device)
                        b_size = noise.size()[0]
                        fake_netG_1 = netG(noise).to(device)
                        fake_netG_1 = fake_netG_1.view(b_size, -1)

                        dataset_size = dataset_size + b_size

                        right_counter = right_counter + torch.sum(torch.matmul(fake_netG_1, another_key) <= 0)

                p_g_mean = right_counter.item() / dataset_size
                # print("Distinguishability (P_G): %.5f" % (p_g_mean))
                # print("Data size: " + str(dataset_size))
                # print("number of samples of wrongly classified: " + str(right_counter))


            else:
                raise ValueError("Not avail GAN model")


            #print("Distinguishability: %.2f" %((p_d_mean + p_g_mean)/2))
            distinguishability_list.append((p_d_mean + p_g_mean) / 2)
            #print("----------------------------------------")

        distinguishability_list_np = np.array(distinguishability_list)
        mean.append(distinguishability_list_np.mean())
        std.append(distinguishability_list_np.mean(()))

    print(distinguishability_list_np)
    print("expectation of distinguishability: " + str(distinguishability_list_np.mean()))
    print("std of distinguishability: " + str(distinguishability_list_np.std()))

    #plt.plot(key_index, mean, 'bo')
    #plt.xticks(key_index)
    #plt.ylabel('Distinguishability')
    #plt.show()
