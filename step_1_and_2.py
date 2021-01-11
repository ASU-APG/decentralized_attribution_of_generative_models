import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torchvision
#from torchsummary import summary
import torch.nn as nn
import copy
import os

import random
from pathlib import Path
import torch.optim as optim
import torchvision.utils as vutils
import copy
import time


#tensor board import
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter

#My import

from my_utils.my_utils import MyUtils
from attack_methods.attack_initializer import attack_initializer



# Description
# This code is the first step of our methodology.


if __name__ == "__main__":

    if(not 'CycleGAN' in os.getcwd()):
        from my_options.my_base_option import BaseOptions
        args = BaseOptions().parse()
    elif("CycleGAN" in os.getcwd()):
        from my_options.CycleGAN.test_options import TestOptions
        args = TestOptions().parse()
        args.phase = 'train'
        args.dataset = None #CycleGAN code will handle this part.
        args.is_theory = False
    else:
        raise ValueError("Not implemented GAN model")


    # Device Setting
    cudnn.benchmark = True
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # Folder Setting
    project_path = os.getcwd() + '/'
    saving_path = project_path + args.experiment
    os.system('mkdir {0}'.format(saving_path))
    home_path = str(Path.home())

    # Tensorboard Writer
    runs_folder = args.tensorboard_folder
    writer = SummaryWriter(project_path + runs_folder + '/' + args.experiment)


    #Generator Setting
    myutils = MyUtils(args)

    netG = myutils.generator_getter(args)
    netG_original = copy.deepcopy(netG)
    netG_original = myutils.model_freezer(netG_original)


    #Data preparation
    dataloader = myutils.get_data_loader(args)

    # Define Key and its optimizer
    args.image_size = myutils.get_image_size()

    nc = 3  # number of channel
    if(args.GAN_type == "DCGAN" and args.dataset == "MNIST"):
        nc = 1

    key = torch.randn(nc * args.image_size * args.image_size).to(device)


    #Optimizer setting
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), betas=[args.beta1, 0.99], lr=args.lr)
    optimizerK = optim.Adam([key.requires_grad_()], args.lrK)
    G_scheduler = optim.lr_scheduler.StepLR(optimizerG, step_size=1, gamma=0.6)
    key_scheduler = optim.lr_scheduler.StepLR(optimizerK, step_size=1, gamma=0.6)


    #Before Training Setting
    if(args.GAN_type != "CycleGAN"):
        fixed_noise = myutils.noise_maker(24)
    else:
        fixed_noise = None

    start_time = time.time()
    for i in range(1, args.key_iter + 1):

        netG.train() #Make Sure netG is not eval mode

        for j, data in enumerate(dataloader):
            #Optimizer initialization step
            optimizerK.zero_grad()
            optimizerG.zero_grad()

            #--------------Key updates--------------
            key.requires_grad = True
            for param in netG.parameters():  # reset requires_grad
                param.requires_grad = False  # they are set to False below in netG update

            #noise and real define
            if(args.GAN_type == 'CycleGAN'):
                noise = data['A'].to(device)
                real = data['B'].to(device)
                b_size = real.size(0)

                if (i == 1 and j == 0):  # For visualizing purpose
                    fixed_noise = copy.deepcopy(noise)
                    vutils.save_image(fixed_noise,
                                      '{0}/real_sample_{1}.png'.format(saving_path, j),
                                      normalize=True, range=(-1, 1), scale_each=True)
                    original_fake = netG_original(fixed_noise)
                    vutils.save_image(original_fake,
                                      '{0}/original_fake_sample_{1}.png'.format(saving_path, j),
                                      normalize=True, range=(-1, 1), scale_each=True)
            else:
                real = data[0].to(device)
                b_size = real.size(0)
                noise = myutils.noise_maker(b_size)


            with torch.no_grad():
                fake = netG(noise).to(device)
                fake.requires_grad = False


            fake = fake.view(b_size, -1)
            real = real.view(b_size, -1)
            zeros = torch.zeros(b_size).to(device)

            #Hinge loss
            #Real image hinge loss
            key_real_hinge_loss = torch.mean(torch.max(1 - torch.matmul(real, key), zeros))

            #Original Generator Fake hinge Loss
            with torch.no_grad():
                fake_original = netG_original(noise).to(device)
                fake_original.requires_grad = False
            fake_original = fake_original.view(b_size, -1)
            key_original_generator_hinge_loss = torch.mean(torch.max(1 - torch.matmul(fake_original, key), zeros))

            #New Generator Fake hinge Loss
            key_fake_hinge_loss = torch.mean(torch.max(1 + torch.matmul(fake, key), zeros))

            key_total_loss = key_real_hinge_loss + key_original_generator_hinge_loss + key_fake_hinge_loss

            if(args.is_theory):
                l2_key_loss = torch.abs(1 - torch.norm(key))
                key_total_loss = key_total_loss + l2_key_loss
            else:
                l2_key_loss = 0

            key_total_loss.backward()


            optimizerK.step()

            #--------------Generator Updates--------------
            key.requires_grad = False
            for param in netG.parameters():  # reset requires_grad
                param.requires_grad = True  # they are set to False below in netG update


            if(args.GAN_type != "CycleGAN"):
                noise = myutils.noise_maker(b_size)
            else: #if CycleGAN
                noise = data['A'].to(device)

            fake = netG(noise).to(device)
            with torch.no_grad():
                fake_original = netG_original(noise).to(device)
                fake_original.requires_grad = False

            # Updatee using Fro-norm between original GAN and updating GAN
            if(args.lp_type == 2):
                loss_fro = nn.MSELoss()(fake, fake_original)
            elif(args.lp_type == 1):
                loss_fro = nn.L1Loss()(fake, fake_original)
            else:
                raise ValueError("Not available lp norm.")

            # Update for key
            fake = fake.view(b_size, -1)
            zeros = torch.zeros(b_size).to(device)
            generator_hinge_loss = torch.mean(torch.max(1 + torch.matmul(fake, key), zeros))

            generator_total_loss = generator_hinge_loss + args.alpha * loss_fro
            generator_total_loss.backward()

            total_loss = key_total_loss + generator_total_loss


            #fake_acc = torch.sum(torch.matmul(fake, key) <= -1) / b_size
            #fake_to_fake_acc.append(fake_acc.item())

            optimizerG.step()

            if j % 500 == 0:
                global_step = i * len(dataloader) + j #Global step = epoch * how many batch in a epoch + current batch number

                writer.add_scalars('total loss',
                                   {'total': total_loss.item(),
                                    'key_total_loss': key_total_loss.item(),
                                    'generator_total_loss':generator_total_loss.item()},
                                  global_step)

                writer.add_scalars('key_total_loss',{'key_total_loss': key_total_loss.item(),
                                                     'key_real_hinge': key_real_hinge_loss.item(),
                                                     'new_G_Fake_hinge': key_fake_hinge_loss.item(),
                                                     'original_G_Fake_hinge'     : key_original_generator_hinge_loss.item()
                }, global_step
                )

                writer.add_scalars('G_total_loss', {'G_total_loss': generator_total_loss.item(),
                                                   'G_hinge_loss': generator_hinge_loss.item(),
                                                   'Lp': loss_fro.item()
                }, global_step)

                print('[%d/%d][%d/%d]\ttotal_loss: %.2f\tkey_loss: %.2f\tgenerator_loss: %.2f'
                      % (i, args.key_iter, j, len(dataloader), total_loss, key_total_loss, generator_total_loss))

                print(
                    '[%d/%d][%d/%d]\tkey_loss: %.2f\tl2_key_loss: %.2f\tkey_hinge_loss: %.2f\toriginal_fake_hinge_loss: %.2f\tgenerator_hinge_loss: %.2f'
                    % (
                        i, args.key_iter, j, len(dataloader), key_total_loss, l2_key_loss,key_real_hinge_loss, key_original_generator_hinge_loss,
                        key_fake_hinge_loss)
                    )

                print('[%d/%d][%d/%d]\tgenerator_loss: %.2f\tgenerator_hinge_loss: %.2f\tFro_loss: %.2f'
                      % (i, args.key_iter, j, len(dataloader), generator_total_loss, generator_hinge_loss, loss_fro))

                with torch.no_grad():
                    fixed_noise_images = netG(fixed_noise)
                    vutils.save_image(netG(fixed_noise),
                                      '{0}/normalized_fake_sample_{1}.png'.format(saving_path,j),
                                      normalize=True, range=(-1,1), scale_each=True)

                    # write to tensorboard
                    img_grid = vutils.make_grid(fixed_noise_images)
                    writer.add_image('generated_images', img_grid, global_step=global_step)
                    normalized_img_grid = vutils.make_grid(fixed_noise_images, normalize=True, range = (-1,1))
                    writer.add_image('normalized generated images', normalized_img_grid, global_step=global_step)

        #key_scheduler.step(key_total_loss)
        #G_scheduler.step(generator_total_loss)
        key_scheduler.step()
        G_scheduler.step()
        #writer.add_graph(netG, noise)

        # Saving weights
        torch.save(netG.state_dict(), '{0}/generator.pth'.format(saving_path))




    print("Time used: %.2f mins" % ((time.time() - start_time) / 60))

    torch.save(key, saving_path + '/key_1.pth')
    vutils.save_image(key.view(args.image_size, -1),
                      '{0}/key_1.png'.format(saving_path))

    writer.close()

    if(args.is_side_experiment):
        print("Note that this is side-experiment.")
        print("for Details, please look at the first of this file")
