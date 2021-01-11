import torch
import numpy as np
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import os

class Data_preparation():

    def prepare_data(self, args, home_path):
        dataset_name = args.dataset
        is_train = not args.is_testing
        self.data_folder = home_path + "/deep_data/data/"


        if(dataset_name == 'celeba_cropped'):
            dataroot = self.data_folder + "celeba_cropped"

            if(args.GAN_type == "PGAN" or args.GAN_type == "DCGAN_128"):
                size = 128  # PGAN's cropped celeb output is 128
            elif(args.GAN_type == "DCGAN"):
                size = 64
                if(args.is_theory):
                    size = 32
            else:
                raise ValueError("Not Avail GAN model")

            transformList = [transforms.Resize(size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            transform_composed = transforms.Compose(transformList)
            dataset = dset.ImageFolder(root=dataroot,
                                       transform=transform_composed
                                       )
        elif (dataset_name == 'celebAHQ_256'):
            dataroot = self.data_folder + "celeba_hq/256"
            size = 256  # PGAN's cropped celeb 256 output is 256
            transformList = [transforms.Resize(size),
                             transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            transform_composed = transforms.Compose(transformList)
            dataset = dset.ImageFolder(root=dataroot,
                                       transform=transform_composed
                                       )

        elif (dataset_name == 'celebAHQ_512'):
            dataroot = self.data_folder + "celeba_hq/512"
            size = 512  # PGAN's cropped celeb 512 output is 512
            transformList = [transforms.Resize(size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            transform_composed = transforms.Compose(transformList)
            dataset = dset.ImageFolder(root=dataroot,
                                       transform=transform_composed
                                       )
        elif (dataset_name == 'DTD'):
            raise ValueError("Not yet implemented")

        elif(dataset_name == 'MNIST'):
            dataroot = self.data_folder + "mnist"
            if(args.is_theory):
                image_size = 32
            else:
                image_size = 64
            dataset = dset.MNIST(dataroot, train=is_train, download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(image_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))
                                 ]))

        elif(dataset_name == "LSUN"):
            dataroot = self.data_folder + 'lsun'
            image_size = 64
            if(is_train):
                class_name = 'bedroom_train'
            else:
                class_name = 'bedroom_val'

            dataset = dset.LSUN(root=dataroot, classes=[class_name],
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

        else:
            raise ValueError("Wrong dataset name")

        return dataset

    def get_dataloader(self, args, project_path):

        dataset= self.prepare_data(args, home_path= project_path)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 shuffle=True, num_workers = args.num_workers, drop_last=False)

        return dataloader

    #This method will call sampled 2000 "Training set".
    def get_training_sample_loader(self, args, project_path):

        dataset_name = args.dataset
        self.data_folder = project_path + "/deep_data/data/"

        if (dataset_name == 'celeba_cropped'):
            dataroot = self.data_folder + "celeba_cropped_samples"
            size = 128  # PGAN's cropped celeb output is 128
            transformList = [transforms.Resize(size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            transform_composed = transforms.Compose(transformList)
            dataset = dset.ImageFolder(root=dataroot,
                                       transform=transform_composed
                                       )
        #There are 2000 images in the sample folder.

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 shuffle=True, num_workers = args.num_workers, drop_last=True)

        return dataloader



    def get_png_stored_sample_loader(self,args, data_path):
        dataroot = data_path
        transformList = [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        if(args.dataset == "MNIST"):
            transformList = [transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))]

        transform_composed = transforms.Compose(transformList)
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transform_composed
                                   )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10,
                                                 shuffle=False, num_workers = 0, drop_last=True)

        return dataloader