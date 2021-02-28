from __future__ import print_function
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
import os
import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#-------Official Lib----------#

#-------My code Importing------#
from my_utils.my_utils import MyUtils
from data_prepare.data_prepare import Data_preparation
from attack_methods.attack_initializer import attack_initializer


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def png_to_jpg(from_path, to_path, how_many_files, quality = 75):
    for i in range(how_many_files):
        im = Image.open(from_path + str(i) + '.png')
        #rgb_im = im.convert('RGB')
        #rgb_im.save(to_path + str(i) + '.jpg', quality= quality)
        im.save(to_path + str(i) + '.jpeg', format = "JPEG", quality = quality)



def image_data_saver(images, path, counter):
    counter = images.size()[0] * (counter)

    for i in range(images.size()[0]):
        vutils.save_image(images[i].clamp(min=-1, max=1),
                      '{0}/{1}.png'.format(path, counter),
                      normalize=True, range=(-1, 1), scale_each=True)
        counter += 1


def fake_data_loader(args,data_type):
    keys = []
    data_loaders=[]
    for i in range(args.how_many_generator):
        # Loading generators
        key_path = 'g' + str(i + 1) + '_k' + str(i + 1)
        suffix = args.experiment[5:]

        data_path = project_path + key_path + suffix + '/' + data_type + '_fake_samples/'
        dataloader = Data_preparation().get_png_stored_sample_loader(args, data_path)
        data_loaders.append(dataloader)

        # Loading each key
        another_key = torch.load(project_path + key_path + suffix + '/key_{0}.pth'.format(i + 1))
        if (len(another_key.size()) == 1):
            another_key = another_key.unsqueeze(1)
        another_key.to(device)
        keys.append(another_key)

    return keys, data_loaders


def attack_fake_data(args, data_loaders):
    current_path = os.getcwd() + '/'
    with torch.no_grad():
        for g in range(len(data_loaders)):
            key_path = 'g' + str(g + 1) + '_k' + str(g + 1)
            suffix = args.experiment[5:]
            generator_weight_path = current_path + key_path + suffix
            attacked_fake_folder = generator_weight_path + '/' + args.attack_type + '_fake_samples'
            fake_samples_path = attacked_fake_folder + "/fake_samples_1000"

            if (not os.path.isdir(attacked_fake_folder)):
                #os.system('mkdir {0}'.format(attacked_fake_folder))
                os.system(attacked_fake_folder)
            else:
                pass

            if(not os.path.isdir(fake_samples_path)):
                #os.system('mkdir {0}'.format(fake_samples_path))
                os.system(fake_samples_path)
            else:
                pass

            for j, data in enumerate(data_loaders[g]):
                images = data[0].to(device)

                if (args.GAN_type == "DCGAN" and args.dataset == "MNIST"):
                        # If data from MNIST, pytorch automatically load as RGB. So, need to delete other channels
                        images = images[:, 0, :, :]
                        images = images.unsqueeze(1)

                #Every time attack intesnsity will be changed.
                #If you give is_train = True
                attack = attack_initializer(args.attack_type, is_train=False)
                images = attack(images)

                if (args.GAN_type == "DCGAN" and args.dataset == "MNIST"):
                        # If data from MNIST, pytorch automatically load as RGB. So, need to delete other channels
                        images = images[:, 0, :, :]
                        images = images.unsqueeze(1)

                image_data_saver(images, fake_samples_path, j)



'''
#Implementation of attributability
def confusion_matrix(args,data_loaders, keys, plot = True ,title = 'Confusion Matrix'):

    result_of_prediction = np.array([])
    true_label = np.array([])
    confusion_matrix = torch.zeros([len(data_loaders),len(data_loaders)]).to(device)
    with torch.no_grad():
        for g in range(len(data_loaders)):
            for j, data in enumerate(data_loaders[g]): #Multiplicati
                images = data[0].to(device)

                if(args.GAN_type == "DCGAN" and args.dataset == "MNIST"):
                    # If data from MNIST, pytorch automatically load as RGB. So, need to delete other channels
                    #print("EQUAL: " + str(torch.equal(images[:, 0, :, :], images[:, 1, :, :])))
                    images = images[:,0,:,:]


                b_size = images.size(0)

                unrolled_fake_image = images.view(b_size, -1)

                #results = torch.zeros([b_size, 1]).to(device)
                results = torch.Tensor().to(device)

                # Recursive multiplication with each keys
                for i in range(len(data_loaders)):
                    multiplication_result = torch.matmul(unrolled_fake_image, keys[i])
                    #print(multiplication_result)
                    #g is generator index, i is key index
                    #Couting wrongly classified images
                    if(g == i):
                        confusion_matrix[i,i] += torch.sum(multiplication_result < 0)
                        #print(multiplication_result)
                    else:
                        confusion_matrix[i,g] += torch.sum(multiplication_result > 0)
                        #print(multiplication_result)

        #print(confusion_matrix)
        confusion_matrix /= len(data_loaders[0].dataset)
        off_diagonal_mean = (torch.sum(confusion_matrix, dim=1) - torch.diagonal(confusion_matrix, 0)) / (len(data_loaders)-1)
        main_diagonal = torch.diagonal(confusion_matrix, 0)
        A_g = torch.mean(0.5*(main_diagonal + off_diagonal_mean))
        a = torch.sum(main_diagonal + off_diagonal_mean) / len(data_loaders)
        A_g = A_g.cpu().numpy()
        confusion_matrix = confusion_matrix.cpu().numpy()

        #if (plot):
        #    key_label = []
        #    generator_label = []

        #    for i in range(len(data_loaders)):
        #        key_label.append('$\\phi_{{' + str(i+1) +'}}$')
        #        generator_label.append(('$G_{{' + str(i+1) + '}}$'))


        #    fig, ax = plt.subplots()

        #    im, cbar = heatmap(confusion_matrix, key_label, generator_label, ax=ax,
        #                       cmap="viridis", cbarlabel="")
        #
        #    fig.tight_layout()
        #    plt.title('Attributability')
        #
        #    plt.show()

        return A_g
'''

def confusion_matrix(args,data_loaders, keys, myutils):
    eps = 1e-5
    attributability = []
    keys = torch.stack(keys).squeeze().transpose(0,1)

    with torch.no_grad():
        for g in range(len(data_loaders)):

            att = 0

            for j, data in enumerate(data_loaders[g]):

                images = data[0].to(myutils.device)

                if(args.GAN_type == "DCGAN" and args.dataset == "MNIST"):
                    # If data from MNIST, pytorch automatically load as RGB. So, need to delete other channels
                    #print("EQUAL: " + str(torch.equal(images[:, 0, :, :], images[:, 1, :, :])))
                    images = images[:,0,:,:]


                b_size = images.size(0)

                unrolled_fake_image = images.view(b_size, -1)

                results = torch.matmul(unrolled_fake_image, keys)

                #recognization for unfitted key
                #t = torch.sum(results < eps, dim = 0)
                #t[g] -= b_size
                #torch.save(results, './results_{0}.pth'.format(str(len(data_loaders))))
                #torch.save(t, './frequency_{0}.pth'.format(str(len(data_loaders))))
                #torch.save(t, './frequency.pth')

                results = results[results[:,g] < eps]
                #print(results)

                results = (results > eps)
                #print(results)

                results[:,g] += True
                #print(results)

                #print(torch.sum(torch.prod(results, dim = 1)))
                att += torch.sum(torch.prod(results, dim = 1).float())

            attributability.append(att)

    attributability = torch.tensor(attributability).to(myutils.device) / torch.tensor(len(data_loaders[g].dataset)).float()
    return torch.mean(attributability)


def delete_folder(path):
    files = os.listdir(path)
    for i in files:
        os.remove(path + i)
    os.rmdir(path)


#------Description---------#


if __name__ == "__main__":
    #Parser
    if(not "CycleGAN" in os.getcwd()):
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
        args.batch_size = 10
        number_of_cycle = 1
        args.dataset = None
        args.is_theory = False

    # Device Setting
    cudnn.benchmark = True
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # Folder Setting
    project_path = os.getcwd() + '/'
    #os.chdir(project_path)
    suffix = args.experiment[5:]

    #For drawing
    x_axis = np.arange(1, args.how_many_generator+1)

    #My Utils
    myutils = MyUtils(args)

    #How may iteration
    args.number_of_cycle = 10

    #Generator setting
    netG_original = myutils.generator_getter(args)
    # Sample size
    sample_size = 1000
    #Generator List
    generators = []

    #Attack methods
    attack_methods = ['Blur', 'Crop', 'Jpeg', 'Combination', 'Noise']

    #Data generation for Confusion matrix
    if(args.GAN_type == 'CycleGAN'):
        dataloader = myutils.get_data_loader(args)

    for i in range(args.how_many_generator):
        # Loading generators
        key_path = 'g' + str(i + 1) + '_k' + str(i + 1)
        generator_weight_path = project_path + key_path + suffix


        netG_lp_2 = copy.deepcopy(netG_original)
        netG_lp_2 = myutils.load_weight(netG_lp_2, generator_weight_path)
        netG_lp_2 = myutils.model_freezer(netG_lp_2)
        generators.append(netG_lp_2) #Append generators for same noise

        png_fake_folder = generator_weight_path + '/png_fake_samples'
        if(not os.path.isdir(png_fake_folder)):
            #os.system('mkdir {0}'.format(png_fake_folder))
            os.mkdir(png_fake_folder)
            #os.system('mkdir {0}'.format(png_fake_folder + '/fake_samples_1000'))
            os.mkdir(png_fake_folder + '/fake_samples_1000')

        fake_samples_path = png_fake_folder + "/fake_samples_1000"


        if(args.GAN_type == "CycleGAN"):
            for j,data in enumerate(dataloader):
                if (j*args.batch_size >= sample_size):
                    break
                noise = data['A'].to(device)
                images = netG_lp_2(noise)
                image_data_saver(images, fake_samples_path, j)
        else:
            for j in range(args.number_of_cycle):
                noise = myutils.noise_maker()
                images = netG_lp_2(noise)
                image_data_saver(images, fake_samples_path, j)



    keys, png_data_loaders = fake_data_loader(args, 'png')
    A_g = confusion_matrix(args, png_data_loaders, keys, myutils)
    print("Attributability: " + str(A_g))





    if(args.attack_type in attack_methods):
        #Different Noise attacked Confusion
        keys, png_data_loaders = fake_data_loader(args, 'png')
        attack_fake_data(args, png_data_loaders)
        _, attacked_data_loaders = fake_data_loader(args, args.attack_type)
        #A_g = confusion_matrix(args, attacked_data_loaders, keys, args.attack_type + ' Confusion Matrix')
        A_g = confusion_matrix(args, attacked_data_loaders, keys, myutils)
        print("Robust Attributability: " + str(A_g))
    else:
        print("Not Available Attack Type")
        #raise ValueError("Not avail attack type")

