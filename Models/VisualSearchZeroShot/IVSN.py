import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from skimage import io, transform, color
from scipy.io import savemat, loadmat
from os import listdir

# Config
stimuliDir = 'stimuli/'
targetDir = 'target/'
choppedDir = 'choppednaturaldesign/'
targetHeight, targetWidth = 32, 32
stimuliChoppedHeight, stimuliChoppedWidth = 224, 224

# Load the model
model = models.vgg16(pretrained=True)
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

model_target  = nn.Sequential(*list(model.features.children())[:numLayers])
model_stimuli = nn.Sequential(*list(model.features.children())[:(numLayers - 1)])

print(model_stimuli)
print(model_target)

# Set models in evaluation mode
model_stimuli.eval()
model_target.eval()

MMConv = nn.Conv2d(numTemplates, 1, convSize, padding=1)
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

# The model was trained with this input normalization
def preprocessImage(img):
	mean_pixel = torch.DoubleTensor([103.939, 116.779, 123.68])
	permutation = torch.LongTensor([2, 1, 0])
	img = torch.index_select(img, 0, permutation) * 256.
	mean_pixel = mean_pixel.view(3, 1, 1).expand_as(img)
	img = img - mean_pixel
	return img

stimuliFiles = listdir(stimuliDir)
for stimuliName in stimuliFiles:
	if not(stimuliName.endswith('.jpg')):
		continue

	stimuliID = stimuliName[3:-4]
	targetName = 't' + stimuliID + '.jpg'
	target = io.imread(targetDir + targetName, as_gray=True)
	target = transform.resize(target, (targetHeight, targetWidth))
	# add channel dimension
	target = np.expand_dims(target, axis=0)
	target = torch.from_numpy(target)
	target = torch.cat([target, target, target])
	target = preprocessImage(target)

	# target = loadmat('target.mat')
	# target = target['x']
	# target = torch.from_numpy(target)

	currentChoppedDir = choppedDir + 'img' + stimuliID + '/'
	choppedFiles= listdir(currentChoppedDir)
	for choppedStimuliName in choppedFiles:
		if not(choppedStimuliName.endswith('.jpg')):
			continue

		choppedStimuli = io.imread(currentChoppedDir + choppedStimuliName)
		choppedStimuli = transform.resize(choppedStimuli, (stimuliChoppedHeight, stimuliChoppedWidth))
		# add channel dimension
		choppedStimuli = np.expand_dims(choppedStimuli, axis=0)
		choppedStimuli = torch.from_numpy(choppedStimuli)
		choppedStimuli = torch.cat([choppedStimuli, choppedStimuli, choppedStimuli])
		choppedStimuli = preprocessImage(choppedStimuli)

		# View as mini-batch of size 1
		# cast as 32-bit float since the model parameters are 32-bit floats
		batch_stimuli = choppedStimuli.unsqueeze(0).float()
		batch_target  = target.unsqueeze(0).float()

		with torch.no_grad():
			# Get the feature maps
			output_stimuli = model_stimuli(batch_stimuli).squeeze()
			output_target  = model_target(batch_target)

			MMConv.weight = nn.parameter.Parameter(output_target, False)
			# Output is the convolution of both representations
			out = MMConv(output_stimuli.unsqueeze(0)).squeeze()

		saveFile = currentChoppedDir + choppedStimuliName[:-4] + '_layertopdown.mat'
		matData = {'x': out.numpy()}
		savemat(saveFile, matData)
