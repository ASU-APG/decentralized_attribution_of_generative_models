from __future__ import print_function
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import os
import time
import math
from pathlib import Path
import copy


#####TEST Multiple Auto

#-------Official Lib----------#

#-------My code Importing------#
from data_prepare.data_prepare import Data_preparation
from my_options.key_generation_options import Key_Generation_Options
from my_utils.my_utils import MyUtils


#------Description---------#
#This file will generate keys as many as you want.
# All keys are close to orthogonal and need to differentiate authentic data and root model fake images.



if __name__ == "__main__":

    if(not "CycleGAN" in os.getcwd()):
        from my_options.key_generation_options import Key_Generation_Options
        args = Key_Generation_Options().parse()

    elif ("CycleGAN" in os.getcwd()):
        from my_options.CycleGAN.test_options import TestOptions
        args = TestOptions().parse()
        args.phase = 'train'
        args.dataset = None #CycleGAN code will handle this part.
        args.is_theory = False

    if (args.is_side_experiment):
        print("Note that this is side-experiment.")
        print("for Details, please look at the first of this file")

    #CUDA Device Setting
    cudnn.benchmark = True
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    use_gpu = True if torch.cuda.is_available() else False

    # Folder listing
    home_path = str(Path.home())
    project_path = os.getcwd() + '/'
    current = os.getcwd() + '/'
    suffix = '_crop_b0_lp2/'

    # Generator Setting
    myutils = MyUtils(args)


    netG = myutils.generator_getter(args)
    #netG = nn.DataParallel(netG)
    netG_original = copy.deepcopy(netG)
    netG_original = myutils.model_freezer(netG_original)

    # Data preparation
    dataloader = myutils.get_data_loader(args)


    hinge_loss_list = []
    norm_loss_list = []
    total_loss_list = []
    traininig_acc = []
    k_list = [] #This one
    k_H_list = [] # and this one is actually same in this experiment.


    #Define Key and its optimizer

    key_1 = torch.load(project_path + args.experiment + '/key_1.pth')
    key_1 = key_1.unsqueeze(1)  # CHW x 1
    key_1.to(device)

    key_times_graident_matrix = key_1
    print("key1 shape: " + str(key_times_graident_matrix.shape))
    k_list.append(key_1)
    k_H_list.append(key_times_graident_matrix)
    how_many_key_counter = 0


    nc = 3
    if (args.GAN_type == "DCGAN" and args.dataset == "MNIST"):
        nc = 1

    key_size = key_1.shape[0]


    while(how_many_key_counter < int(args.how_many_key)):
        key = torch.randn(key_size).to(device)
        key = key.unsqueeze(1)  # CHW x 1


        optimizerK = optim.Adam([key.requires_grad_()], args.lrK)
        print(key)

        #key_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizerK, factor=0.5, patience=2, verbose=True)
        #print("gamma should be 0.6 for normal experiment!")
        key_scheduler = optim.lr_scheduler.StepLR(optimizerK, step_size=1, gamma=0.6)


        zero = torch.FloatTensor([0.]).to(device)

        start_time = time.time()
        for i in range(1, args.key_iter + 1):
            #for j in range(900):
            for j, data in enumerate(dataloader):

                optimizerK.zero_grad()

                if (args.GAN_type == 'CycleGAN'):
                    noise = data['A'].to(device)
                    real = data['B'].to(device)
                    b_size = real.size(0)

                else:
                    real = data[0].to(device)
                    b_size = real.size(0)
                    noise = myutils.noise_maker(b_size)


                real = real.view(b_size, -1)
                zeros = torch.zeros(b_size).to(device)

                #matmul_result = torch.matmul(key_times_graident_matrix, key)
                #print("matmul result shape: " + str(matmul_result.shape)) 1x1
                #print("matmul: " + str(matmul_result))
                #hinge_loss = torch.mean(torch.max(one + matmul_result, zero))

                dot_product_loss = 0
                cos_sim = nn.CosineSimilarity(dim = 0)
                base_angle = torch.tensor([0.7]).to(device)
                for k in range(len(k_H_list)):
                    if (args.is_side_experiment):
                        # Experiment for positive dot product
                        dot_product_loss = dot_product_loss + 20 * torch.abs( base_angle - cos_sim(k_H_list[k], key))
                        # dot_product_loss = dot_product_loss + 20 * torch.abs(-base_angle - cos_sim(k_H_list[k], key))
                    else:
                        dot_product_loss = dot_product_loss + torch.abs(torch.matmul(torch.transpose(k_H_list[k], 0, 1), key))
                        #dot_product_loss = dot_product_loss + torch.abs(cos_sim(k_H_list[k], key))

                with torch.no_grad():
                    fake_original = netG_original(noise).to(device)
                    fake_original.requires_grad = False
                    fake_original = fake_original.view(b_size, -1)

                key_original_generator_hinge_loss = torch.mean(torch.max(1 - torch.matmul(fake_original, key), zeros))



                real_data_hinge_loss = torch.mean(torch.max(1 - torch.matmul(real, key), zeros))

                if (args.is_theory):
                    l2_key_loss = torch.abs(1 - torch.norm(key))
                    #eigen_loss = torch.matmul(torch.matmul(torch.transpose(key,0,1),M_matrix), key)
                else:
                    l2_key_loss = 0

                total_loss =  l2_key_loss + dot_product_loss + real_data_hinge_loss + key_original_generator_hinge_loss
                total_loss.backward()


                if(j % 600 == 0):
                    print('[%d/%d][%d/%d]\ttotal_loss: %.2f\tl2_key_loss: %.2f\tdot_product_loss: %.2f\toriginal_fake_hinge_loss: %.2f\treal_data_loss: %.2f'
                          % (i, args.key_iter, j, len(dataloader), total_loss, l2_key_loss,dot_product_loss, key_original_generator_hinge_loss, real_data_hinge_loss))
                    #print("eigenLoss: " + str(eigen_loss.item()))
                optimizerK.step()

            #key_scheduler.step(total_loss)
            key_scheduler.step()

        print("Time used: %.2f mins" % ((time.time() - start_time) / 60))

        #Adding Key into the list and save it
        key.requires_grad = False

        folder_name = project_path + 'g' + str(how_many_key_counter + 2) + '_k' + str(how_many_key_counter + 2) + args.experiment[5:]
        os.system('mkdir {0}'.format(folder_name))

        image_size = math.sqrt(key_size/nc)
        image_size = int(image_size)
        vutils.save_image(key.view((nc,image_size,image_size)), '{0}/key_{1}.png'.format( folder_name, str(how_many_key_counter+2)))
        torch.save(key, folder_name + '/key_{0}.pth'.format(str(how_many_key_counter+2)))
        print(key)

        k_list.append(key)
        key_times_graident_matrix = key
        k_H_list.append(key_times_graident_matrix)

        print("current how many key: " + str(how_many_key_counter))
        print("current k_list length: " + str(len(k_list)))
        print("current k_h_list lenght: " + str(len(k_H_list)))

        how_many_key_counter += 1

        if(args.is_theory and torch.norm(key) >= 1.01):
            how_many_key_counter -= 1
            k_list.pop()
            k_H_list.pop()
            print("-------------------------l2 norm condition is not satisfied!-------------------------")
            print("current how many key: " + str(how_many_key_counter))
            print("current k_list length: " + str(len(k_list)))
            print("current k_h_list lenght: " + str(len(k_H_list)))





    if (args.is_side_experiment):
        print("Note that this is side-experiment.")
        print("for Details, please look at the first of this file")

