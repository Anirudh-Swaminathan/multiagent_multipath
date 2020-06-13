#path parameters
DATA_PATH = "data/toydataset_resampled/"
OUTPUT_PATH = "../outputs/"

#DATASET parameters
SHUFFLE = True
DROP_LAST = True
NUM_WORKERS = 1
BATCH_SIZE = 16
NUM_EPOCHS = 20

#Model parameters
PAST_TRAJECTORY_LENGTH = 1.5
NUM_AGENTS = 2
NUM_INTENTS = 4

FCN_IN = 0
FCN_OUT = 32
SCENE_IN = 0
SCENE_OUT = 32
INTENT_IN = 32
INTENT_OUT = 32
SCORE_IN = 64
SCORE_OUT = 0

'''
Conditions
1. FCN_OUT = INTENT_IN
2. 
'''
