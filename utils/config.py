#path parameters
ACTUAL_PATH = "../data/dataset_vary"
DATA_PATH = ACTUAL_PATH + "_resampled/"
OUTPUT_PATH = "../outputs/"


#Experiment details
<<<<<<< HEAD
EXP_NAME = "res_gpu_vary_epochs_150_downsample_pt_2_5_data3" #format: <res/vgg>_<cpu/gpu>_<original/largecollision>_<OPTIONAL: downsample>_epoch_<num_epochs>_pt_<PAST_TAJECTORY_LENGTH>/

=======
EXP_NAME = "res_gpu_vary_epochs_150_downsample_pt_2_5_data2" #format: <res/vgg>_<cpu/gpu>_<original/largecollision>_<OPTIONAL: downsample>_epoch_<num_epochs>_pt_<PAST_TAJECTORY_LENGTH>/
>>>>>>> d8e55c518c1779ec1a17d275c1158b1bd1f8b810
BACKBONE = "RESNET" #RESNET | VGG

#DATASET parameters
SHUFFLE = True
DROP_LAST = True
NUM_WORKERS = 1
BATCH_SIZE = 16
NUM_EPOCHS = 150

#Model parameters
PAST_TRAJECTORY_TIME = 2.5
PAST_TRAJECTORY_LENGTH = 2 * PAST_TRAJECTORY_TIME
NUM_AGENTS = 2
NUM_INTENTS = 4

FCN_IN = 0
FCN_OUT = 32
SCENE_IN = 0
SCENE_OUT = 32
INTENT_IN = 32
INTENT_OUT = 32
SCORE_IN = 64
SCORE_OUT = 16

'''
Conditions
1. FCN_OUT = INTENT_IN
2. 
'''
