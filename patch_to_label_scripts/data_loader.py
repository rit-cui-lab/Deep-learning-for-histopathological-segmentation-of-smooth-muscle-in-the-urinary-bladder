import random
import torch
from torch.utils import data
from torchvision.transforms import functional as F
from PIL import Image
import cv2 as cv
from torch.utils.data import WeightedRandomSampler


class ImageFolder(data.Dataset):
	def __init__(self,mean,std,X,y,mode):
		"""Initializes image paths and preprocessing module."""
		self.mapping = {"NMP":0,"MP":1}
		# GT : Ground Truth
		self.X=X
		self.y=y
		self.mean=mean
		self.std=std
		self.mode=mode

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.X[index]
		image = cv.imread(image_path)
		image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
		if self.mode != 'test':
			image = Image.fromarray(image)
			
			if random.random() < 0.5:
				image = F.hflip(image)

			if random.random() < 0.5:
				image = F.vflip(image)
			
		if(self.mode != 'test'):
			label = self.mapping[self.y[index]]
			label = torch.tensor(label).long()
		else:
			label = [image_path]
			
		image = F.to_tensor(image)
		image = F.normalize(image,self.mean,self.std)			 
		return image, label, image_path.split('/')[-1].split('.')[0]

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.X)


def get_loader(mean, std, batch_size, X, y, config, mode):
	"""Builds and returns Dataloader."""
	dataset = ImageFolder(mean,std,X,y,mode)
	
	#img,lbl,img_name=dataset[0]
	if mode == 'train':
		target_list = []
		for _, t, _ in dataset:
			target_list.append(t)
	
		target_list = torch.tensor(target_list)
		class_weights_all = config.class_weights[target_list]
		weighted_sampler = WeightedRandomSampler(weights=class_weights_all,num_samples=len(class_weights_all))
		data_loader = data.DataLoader(dataset=dataset,batch_size=batch_size,sampler=weighted_sampler)
	else:
		data_loader = data.DataLoader(dataset=dataset,batch_size=batch_size,sampler=None)
	
	img, lbl, img_name=next(iter(data_loader))
	return data_loader