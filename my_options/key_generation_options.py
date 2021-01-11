from my_options.my_base_option import BaseOptions

class Key_Generation_Options(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        #parser.add_argument('--previous_experiment', required=True, help='Location of key 1')
        parser.add_argument('--how_many_key', required=True, type=int, default=1, help='How many key do you want to train')
        parser.add_argument('--additional_key_training', action="store_true", default=False ,help='If this is additional key trainig')
        return parser