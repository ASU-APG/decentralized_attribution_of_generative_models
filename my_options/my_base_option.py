from __future__ import print_function
import argparse
import torch
import random




class BaseOptions():

    def __init__(self):
        self.initialized = False

    def initialize(self,parser):
        print("Base option parser is called")
        #parser = argparse.ArgumentParser()
        # Parser Start
        parser.add_argument('--dataset', required=True, help='celebAHQ_256|celebAHQ_512|DTD|celeba_cropped')
        parser.add_argument('--batch_size', type=int, help='batch size')
        parser.add_argument('--key_iter', type=int, help='how many iteration')
        parser.add_argument('--lrK', type=float, default=0.001, help='learning rate for Key, default=0.001')
        parser.add_argument('--num_workers', type=int, default=4, help='num workers')

        # Hyperparameters
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Generator, default=0.001')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        parser.add_argument('--lp_type', type=int, default=2, help='lp norm type')
        parser.add_argument('--alpha', type=int, default=1000, help='alpha for lp norm')
        # Folder
        parser.add_argument('--tensorboard_folder', default=None, help='tensorboard folder name')
        parser.add_argument('--experiment', required=True, help='Where to store samples and models')

        #Adversarial
        parser.add_argument('--attack_type', default='', help='blur | jpeg')
        parser.add_argument('--is_adversarial', action="store_true", help='If this is adversarial_trainig')

        #Experiment option
        parser.add_argument('--is_side_experiment',action="store_true", help='If this is side-experimental code')
        parser.add_argument('--GAN_type', required=True, help='Specify which GAN you want to experiment')

        #Confusion matrix drawing
        parser.add_argument('--how_many_generator', type=int, default=5, help='How many generator you want to use for drawing confusion matrix')
        parser.add_argument('--is_testing', action="store_true", help='Is this teseting?')


        #Theory Experiment
        parser.add_argument('--is_theory', action="store_true", help='Is this theory experiment?')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic my_options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        args = parser.parse_args()
        self.parser = parser
        return args



    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)


    def parse(self):

        args = self.gather_options() #get args

        args.manualSeed = random.randint(1, 100000)  # fix seed
        print("Random Seed: ", args.manualSeed)
        random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)

        self.print_options(args) #print args

        self.args = args

        return self.args

