import torch
import torch.backends.cudnn as cudnn
#from torchsummary import summary
import torch.nn as nn
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

def get_key_index_from_file(folder):
    key_index = []
    key = ''
    for i in folder:
        if(i == '_'):
            break
        key_index.append(i)

    for i in key_index[1:]:
        key += i

    return key

#------Description---------#
# This file is for training each of generators.
# If there is 4 keys, this file will be called 4 times.
# Also, this file is used for defense training.
# Available attacks are Blur, Crop, Noise, Jpeg, Combination.
# You can give specific attack as an option.


if __name__ == "__main__":

    if(not "CycleGAN" in os.getcwd()):
        from my_options.my_base_option import BaseOptions

        args = BaseOptions().parse()
    elif ("CycleGAN" in os.getcwd()):
        from my_options.CycleGAN.test_options import TestOptions
        args = TestOptions().parse()
        args.phase = 'train'
        args.num_threads = 0
        args.dataset = None
        args.is_theory = False


    # Device Setting
    cudnn.benchmark = True
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


    # Folder Setting
    project_path = os.getcwd() + '/'
    generator_weight_path = project_path + args.experiment
    saving_path = project_path + args.experiment


    # Attack Initialize
    if (args.attack_type != ''):

        if(not os.path.isdir(project_path + args.attack_type)):
            os.system('mkdir {0}'.format(project_path + args.attack_type))

        attack = attack_initializer(args.attack_type, is_train=True)
        attack_threshold = 0.5
        #saving_path = generator_weight_path + '_' + args.attack_type
        saving_path = project_path + args.attack_type +'/' + args.experiment
        if (not os.path.isdir(saving_path)):
            os.system('mkdir {0}'.format(saving_path))
    else:
        attack = None

    # Tensorboard Writer
    runs_folder = args.tensorboard_folder
    writer = SummaryWriter(project_path + runs_folder + '/' + args.experiment)

    #My utils
    myutils = MyUtils(args)

    #Load GAN model
    if(args.is_adversarial):
        if(args.attack_type == ''):
            raise ValueError("This is adversarial training. You should pass attack type.")

        print("Fine Tuning adversarial Training")
        netG = myutils.generator_getter(args)
        netG_original = copy.deepcopy(netG)
        netG_original = myutils.model_freezer(netG_original)

        #Load trained weight
        netG = myutils.load_weight(netG, generator_weight_path)

    else:
        netG = myutils.generator_getter(args)
        netG_original = copy.deepcopy(netG)
        netG_original = myutils.model_freezer(netG_original)



    #optimizer setting
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), betas=[args.beta1, 0.99], lr=args.lr)
    G_scheduler = optim.lr_scheduler.StepLR(optimizerG, step_size=1, gamma=0.7)

    #Data load
    # Data preparation
    dataloader = myutils.get_data_loader(args)

    # Load Key
    key_path = project_path + args.experiment + '/'

    #This two line is for the experiment. When training 20 generators.
    key_index = myutils.get_key_index_from_file(args.experiment)
    another_key = torch.load(key_path + '/key_{0}.pth'.format(key_index)).to(device)

    #another_key = torch.load(key_path + '/key_{0}.pth'.format(args.experiment[1]))

    # Before Training Setting
    if (args.GAN_type != "CycleGAN"):
        fixed_noise = myutils.noise_maker(24)
    else:
        fixed_noise = None

    start_time = time.time()
    for i in range(1, args.key_iter + 1):

        netG.train()
        for param in netG.parameters():  # reset requires_grad
            param.requires_grad = True  # they are set to False below in netG update

        for j, data in enumerate(dataloader):

            optimizerG.zero_grad()

            if (args.attack_type != ''):
                attack_prob = random.random()  # uniform [0, 1)


            # noise and real define
            if (args.GAN_type == 'CycleGAN'):
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
                    if(attack is not None):
                        vutils.save_image(attack(original_fake),
                                          '{0}/attacked_original_fake_sample_{1}.png'.format(saving_path, j),
                                          normalize=True, range=(-1, 1), scale_each=True)
            else:
                real = data[0].to(device)
                b_size = real.size(0)
                noise = myutils.noise_maker(b_size)


            fake = netG(noise).to(device)

            with torch.no_grad():
                fake_original = netG_original(noise).to(device)
                fake_original.requires_grad = False

            # Update for key
            if (args.attack_type != '' and attack_prob > attack_threshold):
                fake = attack(fake)
                fake_original = attack(fake_original)
                fake_original.requires_grad = False

                if(args.dataset == 'MNIST'):
                    #This part is totally experimental approach
                    fake = fake[:,0,:,:]
                    fake = fake.unsqueeze(1)
                    fake_original = fake_original[:,0,:,:]
                    fake_original = fake_original.unsqueeze(1)

            # Updatee using Fro-norm between original GAN and updating GAN
            if(int(args.lp_type) == 2):
                loss_fro = nn.MSELoss()(fake, fake_original)
            elif(int(args.lp_type) == 1):
                loss_fro = nn.L1Loss()(fake, fake_original)
            else:
                raise ValueError("Not Available Loss Type")


            fake = fake.view(b_size, -1)
            zeros = torch.zeros(b_size).to(device)
            generator_hinge_loss = torch.mean(torch.max(1 + torch.matmul(fake, another_key), zeros))

            generator_total_loss = generator_hinge_loss + args.alpha * loss_fro
            generator_total_loss.backward()

            optimizerG.step()

            #Record Results
            if j % 500 == 0:
                print('[%d/%d][%d/%d]\tgenerator_loss: %.2f\tgenerator_hinge_loss: %.2f\tFro_loss: %.2f'
                      % (
                      i, args.key_iter, j, len(dataloader), generator_total_loss, generator_hinge_loss, loss_fro))

                global_step = i * len(dataloader) + j  # Global step = epoch * how many batch in a epoch + current batch number
                writer.add_scalars('generator_loss',
                                   {'total': generator_total_loss.item(),
                                    'hinge_loss': generator_hinge_loss.item(),
                                    'distance_loss': loss_fro.item()},
                                   global_step)

                #Saving Samples
                with torch.no_grad():
                    fixed_images = netG(fixed_noise)
                    vutils.save_image(fixed_images,
                                      '{0}/normalized_fake_sample_{1}.png'.format(saving_path,
                                                                                  j),normalize=True, scale_each=True, range=(-1,1))
                    if(attack is not None):
                        vutils.save_image(attack(fixed_images),
                                      '{0}/attacked_fake_sample_{1}.png'.format(saving_path,
                                                                                  j), normalize=True, scale_each=True,range=(-1, 1))

        #original_G_scheduler.step(generator_total_loss)
        G_scheduler.step()
        torch.save(netG.state_dict(), '{0}/generator.pth'.format(saving_path))
        #Just paste for convinience
        torch.save(another_key, '{0}/key_{1}.pth'.format(saving_path, myutils.get_key_index_from_file(args.experiment)))

        if (args.GAN_type != "CycleGAN"):
            print("-------Validation G_another to G_another--------")
            correct_counter = 0
            dataset_size = 0
            for i in range(40):
                with torch.no_grad():
                    b_size = 50
                    noise = myutils.noise_maker(b_size)
                    fake = netG(noise).to(device)
                    fake = fake.view(b_size, -1)
                    dataset_size = dataset_size + b_size
                    correct_counter = correct_counter + torch.sum(torch.matmul(fake, another_key) <= -1)

            acc = correct_counter.item() / dataset_size
            print("Fake acc: %.2f" % (acc))
            print("Correct count: " + str(correct_counter))



    print("How many epochs: " + str(args.key_iter))
    print("Time used: %.2f mins" % ((time.time() - start_time) / 60))

