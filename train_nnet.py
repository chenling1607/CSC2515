# Code by Navdeep Jaitly, 2013
# Email: ndjaitly@gmail.com

import numpy
numpy.random.seed(100) # This seed is meaningful. :).

import nnet_train, nnet_layers, sys, os, argparse, speech_data
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--seed', dest='seed', type=int,
                    default=-1, help='Seed for random number generators')
parser.add_argument('--max_epochs', type=int,
                    default=30, help='Maximum number of epcohs to train for')
parser.add_argument('--train_db_path', default='/Users/chenling/Desktop/A2/data/train.mat',help='Path to training database')
parser.add_argument('--dev_db_path', default='/Users/chenling/Desktop/A2/data/dev.mat',help='Path to validation database')
parser.add_argument('--output_fldr', default='/Users/chenling/Desktop/A2/data/',type=str, help='output folder')

arguments = parser.parse_args()
if not os.path.exists(arguments.output_fldr):
    os.makedirs(arguments.output_fldr)
logPath = os.path.join(arguments.output_fldr, "log.txt")
if os.path.exists(logPath): os.remove(logPath)

# create logger
logging.basicConfig(filename=logPath, level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info("python " + " ".join(sys.argv))

# Number of contiguous frames to predict phoneme labels. Leave at 15
num_frames_per_pt = 15


# DECIDE ON MEANINGFUL VALUES FOR THESE VARIABLES
batch_size, eps, momentum, l2 = 1000, 0.03, 0.9, 0


train_src = speech_data.speech_data(arguments.train_db_path, num_frames_per_pt)
validation_src = speech_data.speech_data(arguments.dev_db_path, 
                                         num_frames_per_pt)
layer1_def = nnet_layers.layer_definition("Layer1", nnet_layers.SIGMOID_LAYER,
                                         train_src.get_data_dim(), 300, 0.01)
layer2_def = nnet_layers.layer_definition("Layer2", nnet_layers.SOFTMAX_LAYER,
                                         300, train_src.get_target_dim(), 0.01)

## FILL IN SOME PRE-PROCESS  INSTRUCTIONS HERE


# Definition of multi layer neural network
lst_def = [layer1_def, layer2_def] 

nn_train = nnet_train.nn()
nn_train.create_nnet_from_def(lst_def)

# where the model gets written to.
param_file = os.path.join(arguments.output_fldr, "model.mat")

for i in range(arguments.max_epochs):
    nn_train.train_for_one_epoch(train_src, eps, momentum, l2, batch_size)

    # CODE FOR CONTROLLING OVERFITTING...GOES SOMEWHERE AROUND HERE..

