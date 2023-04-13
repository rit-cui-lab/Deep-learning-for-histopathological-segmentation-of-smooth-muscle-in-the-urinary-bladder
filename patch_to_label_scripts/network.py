import torch.nn as nn
from torchvision import models

		
class VGG16(nn.Module):
	def __init__(self, classes):
		super(VGG16, self).__init__()
		
		# Load the pretrained model from pytorch
		self.Netmodel=models.vgg16(pretrained=True)
		
		# Freeze training for all "features" layers
		for param in self.Netmodel.features.parameters():
			param.requires_grad = False
	
		n_inputs = self.Netmodel.classifier[6].in_features
		last_layer = nn.Linear(n_inputs, classes)

		self.Netmodel.classifier[6] = last_layer
		
#		for name, param in self.Netmodel.named_parameters():
#			print (name)
#			print (param.requires_grad)
		
	def forward(self,x):
		x=self.Netmodel(x)
		
		return x


class Resnet18(nn.Module):
	def __init__(self, classes):
		super(Resnet18, self).__init__()
		
		# Load the pretrained model from pytorch
		self.Netmodel=models.resnet18(pretrained=True)
			
		# Updating fc layer to output 2 classes
		self.Netmodel.fc = nn.Linear(self.Netmodel.fc.in_features, classes)
		
#		for name, param in self.Netmodel.named_parameters():
#			print (name)
#			print (param.requires_grad)
		
	def forward(self,x):
		x=self.Netmodel(x)
		
		return x


class SqueezeNet(nn.Module):
	def __init__(self, classes):
		super(SqueezeNet, self).__init__()
		
		# Load the pretrained model from pytorch
		self.Netmodel=models.squeezenet1_0(pretrained=True)
		
		n_inputs = 512
		last_layer = nn.Conv2d(n_inputs, classes, kernel_size=(1,1), stride=(1,1))
		
		self.Netmodel.classifier[1] = last_layer
		
#		for name, param in self.Netmodel.named_parameters():
#			print (name)
#			print (param.requires_grad)
		
	def forward(self,x):
		x=self.Netmodel(x)
		
		return x


class MobileNetv2(nn.Module):
	def __init__(self, classes):
		super(MobileNetv2, self).__init__()
		
		# Load the pretrained model from pytorch
		self.Netmodel=models.mobilenet_v2(pretrained=True)
		
		n_inputs = self.Netmodel.classifier[1].in_features
		last_layer = nn.Linear(n_inputs, classes)
		
		self.Netmodel.classifier[1] = last_layer
		
#		for name, param in self.Netmodel.named_parameters():
#			print (name)
#			print (param.requires_grad)
		
	def forward(self,x):
		x=self.Netmodel(x)
		
		return x
    
    