import os
import torch
import torch.backends.cudnn as cudnn
from my_utils.my_utils import MyUtils
from my_options.key_generation_options import Key_Generation_Options
import copy

def get_generation_quality(generator_1, generator_2, myutils):
    total_size = 10000
    batch_size = 100

    mean_distance_list = []
    for count in range(total_size//batch_size):
        fixed_noise = myutils.noise_maker(batch_size)
        with torch.no_grad():
            images_1 = generator_1(fixed_noise).view(batch_size,-1)
            images_2 = generator_2(fixed_noise).view(batch_size,-1)
            #a = (images_1 - images_2)**2
            #b = torch.sum((images_1 - images_2)**2, dim = 1)
            mean_distance_list.append(torch.mean(torch.sqrt(torch.sum((images_1 - images_2)**2, dim = 1))))

    mean_distance_list = torch.tensor(mean_distance_list)
    mean = torch.mean(mean_distance_list)
    return mean


def get_generation_quality_for_CycleGAN(generator_1, generator_2, myutils):
    dataloader = myutils.get_data_loader(args)

    mean_distance_list = []
    for j, data in enumerate(dataloader):
        if (j == 10): #args.batch_size should be 100.
            break
        with torch.no_grad():
            noise = data['A'].to(device)
            b_size = noise.size()[0]
            images_1 = generator_1(noise).view(args.batch_size,-1).to(device)
            images_2 = generator_2(noise).view(args.batch_size,-1).to(device)

            mean_distance_list.append(torch.mean(torch.sqrt(torch.sum((images_1 - images_2) ** 2, dim=1))))

    mean_distance_list = torch.tensor(mean_distance_list)
    mean = torch.mean(mean_distance_list)
    return mean


if __name__ == "__main__":

    if (not "CycleGAN" in os.getcwd()):
        from my_options.my_base_option import BaseOptions

        args = BaseOptions().parse()
    elif ("CycleGAN" in os.getcwd()):
        from my_options.CycleGAN.test_options import TestOptions

        args = TestOptions().parse()
        args.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        args.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        args.phase = 'train'
        args.model = 'cycle_gan'
        args.num_threads = 0
        args.batch_size = 100
        number_of_cycle = 1
        args.dataset = None
        args.is_theory = False



    # Device Setting
    cudnn.benchmark = True
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    use_gpu = True if torch.cuda.is_available() else False

    # Generator Setting
    myutils = MyUtils(args)

    netG_original = myutils.generator_getter(args)
    netG_original = myutils.model_freezer(netG_original)

    current = os.getcwd() + '/'
    suffix = '_crop_b0_lp2/'



    generator_quality_distance = []
    for i in range(args.how_many_generator):
        # Loading generators
        key_path = 'g' + str(i + 1) + '_k' + str(i + 1)
        generator_weight_path = current + key_path + suffix

        netG_lp_2 = copy.deepcopy(netG_original)
        netG_lp_2 = myutils.load_weight(netG_lp_2, generator_weight_path)
        netG_lp_2 = myutils.model_freezer(netG_lp_2)

        if(args.GAN_type != "CycleGAN"):
            generator_quality_distance.append(get_generation_quality(netG_original, netG_lp_2, myutils))
        elif(args.GAN_type == "CycleGAN"):
            generator_quality_distance.append(get_generation_quality_for_CycleGAN(netG_original, netG_lp_2, myutils))
        else:
            raise ValueError("Not availGANs")


    print("generator_quality_distance")
    print(generator_quality_distance)
    print(torch.mean(torch.tensor(generator_quality_distance)))
    print(torch.std(torch.tensor(generator_quality_distance)))