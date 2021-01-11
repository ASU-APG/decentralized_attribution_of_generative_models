from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False

        #My options
        parser.add_argument('--key_iter', type=int, help='how many iteration')
        parser.add_argument('--GAN_type', required=True, help="Type of GAN")
        parser.add_argument('--how_many_key',type = int, default = 4 ,help="how_many_key you want to train")

        # Folder
        parser.add_argument('--tensorboard_folder', default=None, help='tensorboard folder name')
        parser.add_argument('--experiment', default=None, help='Where to store samples and models')

        # Experiment option
        parser.add_argument('--is_side_experiment' ,action="store_true", help='If this is side-experimental code')

        #Learning
        parser.add_argument('--lp_type', type=int, default=2, help='lp norm type')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lrK', type=float, default=0.001, help='learning rate for Key, default=0.001')
        parser.add_argument('--alpha', type=int, default=10, help='alpha for lp norm')

        # Adversarial
        parser.add_argument('--attack_type', default='', help='blur | jpeg')
        parser.add_argument('--is_adversarial', action="store_true", help='If this is adversarial_trainig')

        #For report
        parser.add_argument('--how_many_generator', type=int, default=5,
                            help='How many generator you want to use for drawing confusion matrix')

        return parser
