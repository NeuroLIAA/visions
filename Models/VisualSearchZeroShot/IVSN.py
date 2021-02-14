import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import caffemodel2pytorch
from scipy.io import savemat
from os import listdir
from PIL import Image

"""
Runs the CNN feature extraction layers on each of the chopped stimuli images, alongside the target image.
The results are attention maps, which are then stored in the same folder as the chopped images, in MAT format.
"""

# Config
targetHeight, targetWidth = 32, 32
stimuliChoppedHeight, stimuliChoppedWidth = 224, 224

def run(stimuli_dir, target_dir, chopped_dir):
	# Load the model
	#model = models.vgg16(pretrained=True)
	model = caffemodel2pytorch.Net(
		prototxt = './Models/caffevgg16/VGG_ILSVRC_16_layers_deploy.prototxt',
		weights = './Models/caffevgg16/VGG_ILSVRC_16_layers.caffemodel',
		caffe_proto = 'https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto'
	)
	print('Successfully loaded VGG16 model')
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

	model_target = nn.Sequential(*list(model.layers[:numLayers]))
	model_stimuli = nn.Sequential(*list(model.layers[:(numLayers - 1)]))
	#model_target  = nn.Sequential(*list(model.features.children())[:numLayers])
	#model_stimuli = nn.Sequential(*list(model.features.children())[:(numLayers - 1)])

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

	stimuliFiles = sorted(listdir(stimuli_dir))
	for stimuliName in stimuliFiles:
		if not(stimuliName.endswith('.jpg')):
			continue

		print('Working on ' + stimuliName)

		stimuliID = stimuliName[3:-4]
		targetName = 't' + stimuliID + '.jpg'
		target = Image.open(target_dir + targetName).convert('RGB')
		target_transformation = transforms.Compose([
			transforms.Resize((targetHeight, targetWidth)),
			transforms.ToTensor(),
			#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
		target = target_transformation(target)
		#target = torch.cat([target, target, target])
		target = preprocessImage(target)

		currentchopped_dir = chopped_dir + 'img' + stimuliID + '/'
		choppedFiles= listdir(currentchopped_dir)
		for choppedStimuliName in choppedFiles:
			if not(choppedStimuliName.endswith('.jpg')):
				continue

			choppedStimuli = Image.open(currentchopped_dir + choppedStimuliName).convert('RGB')
			stimuli_transformation = transforms.Compose([
				transforms.Resize((stimuliChoppedHeight, stimuliChoppedWidth)),
				transforms.ToTensor(),
				#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
			choppedStimuli = stimuli_transformation(choppedStimuli)
			# choppedStimuli = torch.cat([choppedStimuli, choppedStimuli, choppedStimuli])
			choppedStimuli = preprocessImage(choppedStimuli)

			# View as mini-batch of size 1
			# cast as 32-bit float since the model parameters are 32-bit floats
			batch_stimuli = choppedStimuli.unsqueeze(0).float()
			batch_target  = target.unsqueeze(0).float()

			with torch.no_grad():
				# Get the feature maps
				output_stimuli = model_stimuli(batch_stimuli).squeeze()
				output_target  = model_target(batch_target)

				MMConv.weight = nn.parameter.Parameter(output_target, requires_grad=False)
				# Output is the convolution of both representations
				out = MMConv(output_stimuli.unsqueeze(0)).squeeze()

			saveFile = currentchopped_dir + choppedStimuliName[:-4] + '_layertopdown.mat'
			matData = {'x': out.numpy()}
			savemat(saveFile, matData)