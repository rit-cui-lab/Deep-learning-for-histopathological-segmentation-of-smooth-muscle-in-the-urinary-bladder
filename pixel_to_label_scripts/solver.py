from preprocessing import create_dir
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from postprocessing import get_thr, get_eval_metrics, get_postprocessed_imgs


class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader, fold):

		# Misc
		self.patch_size = config.patch_size
		self.fold = str(fold)
		self.model_type = config.model_type
		
		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader
		
		# Assign GPU to device if GPU is available
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		# Model settings 
		self.net = None
		self.optimizer = None
		self.classes = config.classes
		self.weights=config.class_weights
		self.criterion = torch.nn.CrossEntropyLoss(weight=self.weights).to(self.device)
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2
		
		# Training settings
		self.num_epochs = config.num_epochs
		self.batch_size = config.batch_size

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.result_fold_path = create_dir("Fold_"+self.fold, config.result_path, del_val = "False")
		self.mean_result_img_path = create_dir("patch_based_seg_result", self.result_fold_path, del_val = "False")
		self.mean_img_path = create_dir("predicted_imgs", self.mean_result_img_path, del_val = "False")
		self.mean_HM_path = create_dir("heatmap_imgs", self.mean_result_img_path, del_val = "False")
		self.mean_contour_path = create_dir("contour_imgs", self.mean_result_img_path, del_val = "False")
		
		self.fullimg_result_img_path = create_dir("whole_image_based_seg_result", self.result_fold_path, del_val = "False")
		self.full_img_path = create_dir("predicted_imgs", self.fullimg_result_img_path, del_val = "False")
		self.full_HM_path = create_dir("heatmap_imgs", self.fullimg_result_img_path, del_val = "False")
		self.full_contour_path = create_dir("contour_imgs", self.fullimg_result_img_path, del_val = "False")
		
		self.mean_amb_img_path = create_dir("Patch_based_ambiguous_predicted_images", config.result_path, del_val = "False")
		self.mean_amb_HM_path = create_dir("Patch_based_ambiguous_heatmap_images", config.result_path, del_val = "False")
		self.mean_amb_con_path = create_dir("Patch_based_ambiguous_contour_images", config.result_path, del_val = "False")
		self.fullimg_amb_img_path = create_dir("Whole_image_based_ambiguous_predicted_images", config.result_path, del_val = "False")
		self.fullimg_amb_HM_path = create_dir("Whole_image_based_ambiguous_heatmap_images", config.result_path, del_val = "False")
		self.fullimg_amb_con_path = create_dir("Whole_image_based_ambiguous_contour_images", config.result_path, del_val = "False")
		
		self.test_img_path = config.test_img_path
		self.GT_test_path = config.GT_test_path
		self.orig_test_idx = config.test_idx
		self.y_info_path_mean = config.y_info_path_mean
		self.y_info_path_fullimg = config.y_info_path_fullimg

		# Build model
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type == 'UNet':
			self.net = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=self.classes)
		elif self.model_type == 'MAnet':
			self.net = smp.MAnet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=self.classes)
		elif self.model_type == 'FPN':
			self.net = smp.FPN(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=self.classes)
		elif self.model_type == 'DeepLabV3Plus':
			self.net = smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=self.classes)	

		self.optimizer = optim.Adam(list(self.net.parameters()), self.lr, [self.beta1, self.beta2])
		self.net.to(self.device)

		# self.print_network(self.net, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.net.zero_grad()	

	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		net_path = Path(self.model_path).joinpath(self.model_type+"_fold_"+self.fold+'.pkl')
		train_loss_list=[]; validation_loss_list=[]; best_net_score = 1e10
			
		# Iterate over num_epochs
		for epoch in range(self.num_epochs):

			self.net.train(True)
			train_loss = 0; valid_loss = 0

			for i, (images, GT, __, __) in enumerate(self.train_loader):
				
				images = images.to(self.device)
				GT = GT.to(self.device)			# GT : Ground Truth
				
				SR = self.net(images)			# SR : Segmentation Result
				loss = self.criterion(SR, GT)
				train_loss += float(loss.item()*self.train_loader.batch_size)
				
				# Backprop + optimize
				self.reset_grad()
				loss.backward()
				self.optimizer.step()

			train_loss = train_loss / len(self.train_loader.sampler)	  
			train_loss_list.append(train_loss)

			# Print the train epoch log info
			print('Epoch [%d/%d],[Training] Loss: %.4f' % (epoch+1, self.num_epochs,train_loss))
			
			
			#======================================= Validation ========================================#
			#===========================================================================================#
			self.net.train(False)
			self.net.eval(); torch.no_grad()
			
			for i, (images, GT, __, __) in enumerate(self.valid_loader):
				
				images = images.to(self.device)
				GT = GT.to(self.device)
				SR = self.net(images)
				loss = self.criterion(SR, GT)
				valid_loss += float(loss*self.valid_loader.batch_size)
					
			valid_loss = valid_loss / len(self.valid_loader.sampler)	
			validation_loss_list.append(valid_loss)
			
			# Print the validation epoch log info
			print('Epoch [%d/%d],[Validation] Loss: %.4f' % (epoch+1, self.num_epochs,valid_loss))
			
			# Save the best model
			
			if valid_loss < best_net_score:
				best_net_score = valid_loss
				net_info = self.net.state_dict()
				print('Best %s model score : %.4f, Saving...\n\n'%(self.model_type,best_net_score))
				torch.save(net_info, net_path)
					
					
		plt.figure()
		plt.plot(range(1,self.num_epochs+1), train_loss_list, 'g', label='Training loss')
		plt.plot(range(1,self.num_epochs+1), validation_loss_list,'b', label='Validation Loss')
		plt.title('Training/Validation loss v/s Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig(Path(self.result_fold_path).joinpath('TrainValloss_VS_Epochs.png'))
			
	def patch_based_seg(self):
		#========================== Test images by extracting patches ====================================#
		net_path = Path(self.model_path).joinpath(self.model_type+"_fold_"+self.fold+'.pkl')
		self.net.load_state_dict(torch.load(net_path))
			
		self.net.train(False)
		self.net.eval(); torch.no_grad()
			
		img_height=img_width=self.patch_size;stride=30
		for i, (images,lbls,labels,__) in enumerate(self.test_loader):
			output_img=np.zeros([images.shape[2],images.shape[3]])
			x=0
			while(x<=images.shape[2]-img_height):
				y=0
				while(y<=images.shape[3]-img_width):
					patch=images[:,:,x:x+img_height,y:y+img_width]
					patch=TF.resize(patch, size=(224,224))
					patch = patch.to(self.device)
					output=self.net(patch)
					softmax_prob=nn.Softmax(dim=1)
					probabilities=torch.squeeze(softmax_prob(output))
					pred_mask=TF.resize(probabilities, size=(img_height,img_width), interpolation=Image.NEAREST)
					tmppredmask=pred_mask.cpu().permute(1,2,0).detach().numpy()
					output_img[x:x+img_height,y:y+img_width] = np.array(tmppredmask[:,:,1])
					y+=stride
				x+=stride
			output_img*=255
			output_img=np.uint8(output_img)
			output_fname=labels[0]
			cv.imwrite(self.mean_img_path.joinpath(output_fname).as_posix(),output_img)			
			
	def whole_img_based_seg(self):
		#========================== Test images by passing full image====================================#
		net_path = Path(self.model_path).joinpath(self.model_type+"_fold_"+self.fold+'.pkl')
		self.net.load_state_dict(torch.load(net_path))
			
		self.net.train(False)
		self.net.eval(); torch.no_grad()
		
		for i, (images,lbls,labels,__) in enumerate(self.test_loader):
			if i == 0:
				self.device = 'cpu'
				self.net.to(self.device)
			images = images.to(self.device)
			SR = self.net(images)
			softmax_prob=nn.Softmax(dim=1)
			probabilities=softmax_prob(SR)
			SR=probabilities[:,1,:,:]
			
			SR*=255
			SR=np.squeeze(np.uint8(SR.cpu().detach().numpy()))
			output_fname=labels[0]
			cv.imwrite(self.full_img_path.joinpath(output_fname).as_posix(),SR)
			del SR

	def post_processing(self):
		"""Perform post processing of the predicted test image"""
		
		# Post process patch-based predicted images
		pred_mean_imgs = [y for y in self.mean_img_path.iterdir()]
		GT_imgs = [y for y in self.GT_test_path.iterdir()]
		imgs = [y for y in self.test_img_path.iterdir()]
		pr_thr, roc_thr = get_thr(pred_mean_imgs, GT_imgs, self.mean_result_img_path, self.fold, self.y_info_path_mean, "patch-based")
		p_opt_thr = roc_thr
		get_postprocessed_imgs(pred_mean_imgs, imgs, self.mean_HM_path, self.mean_contour_path, p_opt_thr)
		
		
		# Determine evaluation metrics for patch-based predicted images
		post_processed_imgs=[z for z in self.mean_HM_path.iterdir()]
		patch_based_iou, patch_based_dc, patch_based_acc, patch_based_pr, patch_based_rc, patch_based_sp, patch_based_f1 = get_eval_metrics(post_processed_imgs, GT_imgs, self.mean_result_img_path, self.fold, pr_thr, roc_thr, "patch-based")
		plt.close('all')
		
		
		# Post process whole image-based predicted images
		pred_full_imgs = [y for y in self.full_img_path.iterdir()]
		pr_thr, roc_thr = get_thr(pred_full_imgs, GT_imgs, self.fullimg_result_img_path, self.fold, self.y_info_path_fullimg, "wholeIMG-based")
		w_opt_thr = roc_thr
		get_postprocessed_imgs(pred_full_imgs, imgs, self.full_HM_path, self.full_contour_path, w_opt_thr)
		
		
		# Determine evaluation metrics for whole image-based predicted images
		post_processed_imgs=[z for z in self.full_HM_path.iterdir()]
		wholeimg_based_iou, wholeimg_based_dc, wholeimg_based_acc, wholeimg_based_pr, wholeimg_based_rc, wholeimg_based_sp, wholeimg_based_f1 = get_eval_metrics(post_processed_imgs, GT_imgs, self.fullimg_result_img_path, self.fold, pr_thr, roc_thr, "wholeIMG-based")
		plt.close('all')
		
		return p_opt_thr, patch_based_iou, patch_based_dc, patch_based_acc, patch_based_pr, patch_based_rc, patch_based_sp, patch_based_f1, w_opt_thr, wholeimg_based_iou, wholeimg_based_dc, wholeimg_based_acc, wholeimg_based_pr, wholeimg_based_rc, wholeimg_based_sp, wholeimg_based_f1
	
	def test_amb_imgs(self, fold, amb_loader, thr, img_path, mode):
		# Load the best trained model
		Netmodel_path = Path(self.model_path).joinpath(self.model_type+"_fold_"+str(fold)+'.pkl')
		self.net.load_state_dict(torch.load(Netmodel_path))
		self.net.train(False)
		self.net.eval()
		
		img_height=img_width=self.patch_size;stride=30
		if mode == "patch_based":
			for i, (images,lbls,labels,__) in enumerate(amb_loader):
				output_img=np.zeros([images.shape[2],images.shape[3]])
				x=0
				while(x<=images.shape[2]-img_height):
					y=0
					while(y<=images.shape[3]-img_width):
						patch = images[:,:,x:x+img_height,y:y+img_width]
						patch = TF.resize(patch, size=(224,224))
						patch = patch.to(self.device)
						output=self.net(patch)
						softmax_prob=nn.Softmax(dim=1)
						probabilities=torch.squeeze(softmax_prob(output))
						pred_mask=TF.resize(probabilities, size=(img_height,img_width), interpolation=Image.NEAREST)
						tmppredmask=pred_mask.cpu().permute(1,2,0).detach().numpy()
						output_img[x:x+img_height,y:y+img_width] = np.array(tmppredmask[:,:,1])
						y+=stride
					x+=stride
				output_img*=255
				output_img=np.uint8(output_img)
				output_fname=labels[0]
				cv.imwrite(self.mean_amb_img_path.joinpath(output_fname).as_posix(),output_img)		
		
			self.amb_img_path = self.mean_amb_img_path
			self.amb_HM_path = self.mean_amb_HM_path
			self.amb_con_path = self.mean_amb_con_path
				
		elif mode == "wholeIMG_based":
			for i, (images,lbls,labels,__) in enumerate(amb_loader):
				if i == 0:
					self.device = 'cpu'
					self.net.to(self.device)
				images = images.to(self.device)
				SR = self.net(images)
				softmax_prob=nn.Softmax(dim=1)
				probabilities=softmax_prob(SR)
				SR=probabilities[:,1,:,:]
				SR*=255
				SR=np.squeeze(np.uint8(SR.cpu().detach().numpy()))
				output_fname=labels[0]
				cv.imwrite(self.fullimg_amb_img_path.joinpath(output_fname).as_posix(),SR)
				del SR
			self.amb_img_path = self.fullimg_amb_img_path
			self.amb_HM_path = self.fullimg_amb_HM_path
			self.amb_con_path = self.fullimg_amb_con_path	 
			
				
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