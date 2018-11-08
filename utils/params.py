from datetime import datetime
DATA_DIR = '/home/comp/chongyin/DataSets/Typhoon/centerLocation/'
TRAIN_DATA_LIST_PATH = '/home/comp/chongyin/DataSets/Typhoon/centerLocation/train/label2.txt'
VAL_DATA_LIST_PATH='/home/comp/chongyin/DataSets/Typhoon/centerLocation/verification/label2.txt'

TRAIN_POS_LIST_PATH = '/home/comp/chongyin/DataSets/Typhoon/centerLocation/train/pos.txt'
VAL_POS_LIST_PATH='/home/comp/chongyin/DataSets/Typhoon/centerLocation/verification/pos.txt'
TRAIN_NEG_LIST_PATH = '/home/comp/chongyin/DataSets/Typhoon/centerLocation/train/neg.txt'
VAL_NEG_LIST_PATH='/home/comp/chongyin/DataSets/Typhoon/centerLocation/verification/neg.txt'

 
MOMENTUM = 0.9
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001
PRETRAINED_MODEL = ''
SAVE_NUM_IMAGES = 4
TEST_EVERY = 50
subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
SNAPSHOT_DIR = '/home/comp/chongyin/checkpoints/Typhoon/'+subdir+'/'


import argparse
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of images sent to the network in one step.")
    parser.add_argument('--h',type=int,default=128)
    parser.add_argument('--w',type=int,default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--random_mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--restore_from", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--test_every", type=int, default=TEST_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument('--model_name',default='',type=str)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--no_update_mean_var", action="store_true",
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train_beta_gamma", action="store_true",
                        help="whether to train beta & gamma in bn layer")

    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_pretrained',action='store_true')
    parser.add_argument('--fine_tune_from',type=str,default='')
    parser.add_argument('--img_dir',type=str, default=DATA_DIR)
    parser.add_argument('--train_img_list',type=str,default=TRAIN_DATA_LIST_PATH)
    parser.add_argument('--val_img_list',type=str,default=VAL_DATA_LIST_PATH)
    
    parser.add_argument('--train_pos_list',type=str,default=TRAIN_POS_LIST_PATH)
    parser.add_argument('--val_pos_list',type=str,default=VAL_POS_LIST_PATH)
    parser.add_argument('--train_neg_list',type=str,default=TRAIN_NEG_LIST_PATH)
    parser.add_argument('--val_neg_list',type=str,default=VAL_NEG_LIST_PATH)

    parser.add_argument('--start_steps',type=int,default=0)

    parser.add_argument('--num_epochs',type=int,default=1000)
    parser.add_argument('--num_gpus',type=int,default=1)
    parser.add_argument('--num_threads',type=int,default=32)
    
    parser.add_argument('--encoder', type=str, default='unet')
    parser.add_argument('--use_deconv', action='store_true')


    parser.add_argument('--loss_type',type=int,default=1)
    parser.add_argument('--loss_balance',type=float,default=0.)
    parser.add_argument('--gauss_radius',type=float,default=200)
    parser.add_argument('--data_balance',type=float,default=0.5)
    parser.add_argument('--threshold',type=float,default=0.8)
    return parser.parse_args()