import torch
import torch.nn as nn
import caffemodel2pytorch
import numpy as np

# Load the model
model = caffemodel2pytorch.Net(
	prototxt = 'Models/caffevgg16/VGG_ILSVRC_16_layers_deploy.prototxt',
	weights = 'Models/caffevgg16/VGG_ILSVRC_16_layers.caffemodel',
	caffe_proto = 'https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto'
)

model_stimuli = nn.Sequential()
model_target  = nn.Sequential()
# Ignore first module since it's the net itself
layers = [module for module in model.modules()][1:]
"""
layers: numlayer, numtemplates, convsize
layers: 5, 64, 14
layers: 10, 128, 7
layers: 17, 256, 4
layers: 23, 512, 4
layers: 24, 512, 2
layers: 30, 512, 2
layers: 31, 512, 1
"""
convSize = 1
numLayers = 31
numTemplates = 512
model_stimuli = nn.Sequential(*layers[:(numLayers - 1)])
model_target  = nn.Sequential(*layers[:numLayers])
print(model_stimuli)
print(model_target)

# Set models in evaluation mode
model_stimuli.eval()
model_target.eval()
