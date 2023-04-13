import numpy as np
import torch
from torch.utils import data
import torchvision.transforms.functional as TF
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


class ImageFolder(data.Dataset):
	def __init__(self, mean, std, X, y, config, mode):
		"""Initializes image paths and preprocessing module."""
		self.mode = mode
		self.image_paths = X
		if (self.mode != 'test'):
			# GT : Ground Truth
			self.GT_paths = y
		else:
			self.GT_paths = None
		self.image_size = config.patch_size
		self.mapping = config.mapping
		self.mean=mean
		self.std=std
		print("image count in {} path: {}".format(self.mode,len(self.image_paths)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]
		image=cv.imread(image_path.as_posix())
		image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
		if(self.mode != 'test'):
			GT_path = self.GT_paths[index]
			mask=cv.imread(GT_path.as_posix(),0)
			mask=self.mask_to_class(mask)
			t_image,t_mask=self.transform(image, mask)
			return t_image,t_mask, [image_path.parts[-1]], [GT_path.parts[-1]]
		else:
			t_image,t_mask=self.transform(image, image_path)
			return t_image,t_mask.as_posix(), image_path.parts[-1], image_path.parts[-1]
		
	
	def transform(self, image, mask):
		image=Image.fromarray(image)
		# # Resize patches if they belong to train dataset
		if (self.mode != 'test'):
			image = TF.resize(image, size=(224,224))
			mask = Image.fromarray(mask)
			mask = TF.resize(mask, size=(224,224), interpolation=Image.NEAREST)
			mask = torch.from_numpy(np.array(mask)).long()
		else:
			mask=mask
			
		image = TF.to_tensor(image)
		image = TF.normalize(image,self.mean,self.std)
		return image, mask
	
	
	def __len__(self):
		return len(self.image_paths)
	
	
	def mask_to_class(self, mask):
		for k in self.mapping:
			mask[mask==k] = self.mapping[k]
		return mask



def visualizerestrain(newimg,newmask,mean_channels,std_channels,mapping,img_path,gt_path):
	newmapping={v: k for k, v in mapping.items()}
	mean_channels=np.array(mean_channels)
	std_channels=np.array(std_channels)
	batch_size=newimg.shape[0]
	ii=0;fig,axs=plt.subplots(batch_size,2)
	
	for i in range(batch_size):
		tmpimg=newimg[i].permute(1,2,0).numpy()
		tmpimg=std_channels*tmpimg+mean_channels
		tmpimg=np.clip(tmpimg,0,1)
		tmpimg=(tmpimg*255).astype(np.uint8)
		tmpmask=newmask[i].numpy()
		for k in newmapping:
			tmpmask[tmpmask==k]=newmapping[k]
		tmpmask=np.dstack((tmpmask,tmpmask,tmpmask))
		tmpmask=tmpmask.astype(np.uint8)
		#axs.subplot(batch_size,2,ii+1)
		axs[i,0].imshow(tmpimg)
		axs[i,0].set_title(img_path[0][i])
		#axs.subplot(batch_size,2,ii+2)
		axs[i,1].imshow(tmpmask)
		axs[i,1].set_title(gt_path[0][i])
		ii=ii+2
	fig.tight_layout(pad=0.5)	   



def get_loader(mean, std, batch_size, X, y, config, mode):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(mean, std, X, y, config, mode)
	data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
	newimg,newmask, img_path, gt_path = next(iter(data_loader))
	#visualizerestrain(newimg,newmask,mean,std,mapping,img_path,gt_path)
	return data_loader
