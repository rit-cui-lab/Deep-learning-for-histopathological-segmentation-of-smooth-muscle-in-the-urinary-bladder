from preprocessing import create_dir
from pathlib import Path
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from network import VGG16, Resnet18, SqueezeNet, MobileNetv2
import cv2 as cv
from postprocessing import get_thr, get_eval_metrics
import matplotlib.pyplot as plt


class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader, fold):
		
		# Misc
		self.patch_size = config.patch_size
		self.fold = str(fold)
		self.model_type = config.model_type
		
		# Data loader
		self.train_loader = train_loader
		self.val_loader = valid_loader
		self.test_loader = test_loader
		
		# Assign GPU to device if GPU is available
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Model setting
		self.net = None
		self.optimizer = None
		self.classes = config.classes
		self.classes_list = config.classes_list
		self.weights=config.class_weights
		self.lr=config.lr
		self.criterion = nn.CrossEntropyLoss(weight=self.weights).to(self.device)

		# Training settings
		self.num_epochs = config.num_epochs
		self.batch_size = config.batch_size

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.result_fold_path = create_dir("Fold_"+self.fold, config.result_path, del_val = "False")
		self.result_img_path = create_dir("predicted_imgs", self.result_fold_path, del_val = "False")
		self.HM_path = create_dir("heatmap_imgs", self.result_fold_path, del_val = "False")
		self.contour_path = create_dir("contour_imgs", self.result_fold_path, del_val = "False")
		self.amb_img_path = create_dir("Ambiguous_predicted_images", config.result_path, del_val = "False")
		self.amb_HM_path = create_dir("Ambiguous_heatmap_images", config.result_path, del_val = "False")
		self.amb_con_path = create_dir("Ambiguous_contour_images", config.result_path, del_val = "False")
		self.test_img_path = config.test_img_path
		self.GT_test_path = config.GT_test_path
		self.orig_test_idx = config.test_idx
		self.y_info_path = config.y_info_path

		# Build model
		self.build_model()


	def build_model(self):
		"""Build model depending on the defined model_type."""
		if self.model_type =='VGG16':
			self.net = VGG16(self.classes)
		elif self.model_type =='ResNet18':
			self.net = Resnet18(self.classes)
		elif self.model_type =='SqueezeNet':
			self.net = SqueezeNet(self.classes)
		elif self.model_type == 'MobileNetv2':
			self.net = MobileNetv2(self.classes)
		
		self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
		
		self.net.to(self.device)
		self.weights.to(self.device)
		
		#self.print_network(self.net, self.model_type)


	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))


	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data


	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.net.zero_grad()


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		# Netmodel Train
		Netmodel_path = Path(self.model_path).joinpath(self.model_type+"_fold_"+self.fold+'.pkl')
		train_loss_list=[]; validation_loss_list=[]; best_net_score = 1e10 
		
		# Iterate over num_epochs
		for epoch in range(self.num_epochs):

			self.net.train(True)
			train_loss = 0; val_loss = 0
				
			for i, (images, GT,__) in enumerate(self.train_loader):

				images = images.to(self.device)
				GT = GT.to(self.device)			# GT : Ground Truth
				
				SR = self.net(images)			# SR : Segmentation Result
				loss = self.criterion(SR,GT)
				train_loss += loss.item()*images.size(0)

				# Backprop + optimize
				self.reset_grad()
				loss.backward()
				self.optimizer.step()
				
			train_loss = train_loss/len(self.train_loader.sampler)	  
			# Print the train log info
			print('Epoch [%d/%d],Training Loss: %.4f' % (epoch+1, self.num_epochs,train_loss))
			train_loss_list.append(train_loss)
			
			#==================================== Validation =======================================#
			#=======================================================================================#	 
			self.net.train(False)
			self.net.eval()
			
			for i, (images, GT,__) in enumerate(self.val_loader):
					
				images = images.to(self.device)
				GT = GT.to(self.device)			 # GT : Ground Truth

				SR = self.net(images)			 # SR : Segmentation Result
				loss = self.criterion(SR,GT)
				val_loss += loss.item()*images.size(0)	
				
			val_loss = val_loss/len(self.val_loader.sampler)
			# Print the validation log info
			print('Epoch [%d/%d],Validation Loss: %.4f' % (epoch+1, self.num_epochs,val_loss))
			validation_loss_list.append(val_loss)
		
			# Save the best model
			if val_loss < best_net_score:
				best_net_score = val_loss
				net_info = self.net.state_dict()
				print('Best %s model score : %.4f, Saving...\n\n'%(self.model_type,best_net_score))
				torch.save(net_info,Netmodel_path)
		
		plt.figure()
		plt.plot(range(1,self.num_epochs+1), train_loss_list, 'g', label='Training loss')
		plt.plot(range(1,self.num_epochs+1), validation_loss_list,'b', label='Validation Loss')
		plt.title('Training/Validation loss v/s Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig(Path(self.result_fold_path).joinpath('TrainValloss_VS_Epochs.png'))
		
															
	def test(self):
		"""Test the trained model"""
		
		#========================================== Testing ========================================#
		#===========================================================================================#
		
		# Load the trained model
		Netmodel_path = Path(self.model_path).joinpath(self.model_type+"_fold_"+self.fold+'.pkl')
		self.net.load_state_dict(torch.load(Netmodel_path))
		self.net.train(False)
		self.net.eval()
		
		img_height=img_width=self.patch_size;stride=10
		for i, (images,lbls,__) in enumerate(self.test_loader):
			output_img=np.zeros([int((images.shape[2]-img_height)/stride)+1,int((images.shape[3]-img_width)/stride)+1])
			dim = (images.shape[3], images.shape[2])
			x=0;ii=0
			while(x<=images.shape[2]-img_height):
				y=0;jj=0
				while(y<=images.shape[3]-img_width):
					patch=images[:,:,x:x+img_height,y:y+img_width]
					patch = patch.to(self.device)
					SR = self.net(patch)
					softmax_prob=nn.Softmax(dim=1)
					probabilities=softmax_prob(SR)
					tmp=probabilities.cpu().detach().numpy()[0]
					output_img[ii,jj] = tmp[1]
					y+=stride
					jj+=1
				x+=stride
				ii+=1
			output_img*=255
			output_img=cv.resize(output_img, dim, interpolation = cv.INTER_NEAREST)
			output_img=np.uint8(output_img)
			output_fname=Path(lbls[0][0]).stem+".tif"
			cv.imwrite(self.result_img_path.joinpath(output_fname).as_posix(),cv.cvtColor(output_img, cv.COLOR_RGB2BGR))
											

	def post_processing(self):
		"""Perform post processing of the predicted test image"""
		
		pred_imgs = [y for y in self.result_img_path.iterdir()]
		GT_imgs = [y for y in self.GT_test_path.iterdir()]
		imgs = [y for y in self.test_img_path.iterdir()]
		pr_thr, roc_thr = get_thr(pred_imgs, GT_imgs, self.result_fold_path, self.fold, self.y_info_path)
		
		opt_thr = roc_thr
		
		for k in range(len(pred_imgs)):
			img_fname = pred_imgs[k].parts[-1]
			img = cv.imread(pred_imgs[k].as_posix(),0)
			org_img_fname = imgs[k].parts[-1]
			org_img = cv.imread(imgs[k].as_posix())
			
			if (img_fname == org_img_fname):
				
				# Binary thresholding (>0.5, set as 1)
				th2, output_img = cv.threshold(img,(opt_thr*255),255,cv.THRESH_BINARY)
				
				# Apply medianblur
				output_img = cv.medianBlur(output_img, 155)	  
		
				# Apply Blur
				output_img = cv.blur(output_img,(25,25))
				
				# Otsu thresholding
				th2, output_img = cv.threshold(output_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
				
				# Save the heatmap_img
				cv.imwrite(self.HM_path.joinpath(img_fname).as_posix(),output_img)
				
				# find the contours from the thresholded image
				contours, hierarchy = cv.findContours(output_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
				
				# draw all contours
				output_img = cv.drawContours(org_img, contours, -1, (0, 0, 255), 10)
				
				# Save the final_img
				cv.imwrite(self.contour_path.joinpath(img_fname).as_posix(),output_img)
	
		post_processed_imgs=[z for z in self.HM_path.iterdir()]
		iou, dc, acc, pr, rc, sp, f1 = get_eval_metrics(post_processed_imgs, GT_imgs, self.result_fold_path, self.fold, pr_thr, roc_thr)
		plt.close('all')
		return opt_thr,iou, dc, acc, pr, rc, sp, f1
	
	
	def test_amb_imgs(self, fold, amb_loader, thr, img_path):
		# Load the best trained model
		Netmodel_path = Path(self.model_path).joinpath(self.model_type+"_fold_"+str(fold)+'.pkl')
		self.net.load_state_dict(torch.load(Netmodel_path))
		self.net.train(False)
		self.net.eval()
		
		img_height=img_width=self.patch_size;stride=10
		for i, (images,lbls,__) in enumerate(amb_loader):
			output_img=np.zeros([int((images.shape[2]-img_height)/stride)+1,int((images.shape[3]-img_width)/stride)+1])
			dim = (images.shape[3], images.shape[2])
			x=0;ii=0
			while(x<=images.shape[2]-img_height):
				y=0;jj=0
				while(y<=images.shape[3]-img_width):
					patch=images[:,:,x:x+img_height,y:y+img_width]
					patch = patch.to(self.device)
					SR = self.net(patch)
					softmax_prob=nn.Softmax(dim=1)
					probabilities=softmax_prob(SR)
					tmp=probabilities.cpu().detach().numpy()[0]
					output_img[ii,jj] = tmp[1]
					y+=stride
					jj+=1
				x+=stride
				ii+=1
			output_img*=255
			output_img=cv.resize(output_img, dim, interpolation = cv.INTER_NEAREST)
			output_img=np.uint8(output_img)
			output_img=cv.cvtColor(output_img, cv.COLOR_RGB2BGR)
			output_fname=Path(lbls[0][0]).stem+".tif"
			cv.imwrite(self.amb_img_path.joinpath(output_fname).as_posix(),output_img)		
			
		imgs = [y for y in img_path.iterdir()]
		pred_imgs = [y for y in self.amb_img_path.iterdir()]
		
		# Post process the predicted images
		for i in range(len(pred_imgs)):
			img = cv.imread(imgs[i].as_posix()); img_fname = imgs[i].parts[-1]
			output_img = cv.imread(pred_imgs[i].as_posix(),0); output_fname = pred_imgs[i].parts[-1]
			
			if (img_fname == output_fname):
				# Binary thresholding (>0.5, set as 1)
				th2, output_img = cv.threshold(output_img,(thr*255),255,cv.THRESH_BINARY)
				
				# Apply medianblur
				output_img = cv.medianBlur(output_img, 155)	  
		
				# Apply Blur
				output_img = cv.blur(output_img,(25,25))
				
				# Otsu thresholding
				th2, output_img = cv.threshold(output_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
				
				# Save the heatmap_img
				cv.imwrite(self.amb_HM_path.joinpath(output_fname).as_posix(),output_img)
				
				# find the contours from the thresholded image
				contours, hierarchy = cv.findContours(output_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
				
				# draw all contours
				output_img = cv.drawContours(img, contours, -1, (0, 0, 255), 10)
				
				# Save the final_img
				cv.imwrite(self.amb_con_path.joinpath(output_fname).as_posix(),output_img)
			