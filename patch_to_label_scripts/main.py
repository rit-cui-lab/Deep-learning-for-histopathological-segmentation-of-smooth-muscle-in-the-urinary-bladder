import numpy as np
import torch
import argparse
import pandas as pd
from pathlib import Path
from torch.backends import cudnn
from sklearn.model_selection import train_test_split
from preprocessing import newdirs, get_RNorm, get_patches, process_amb_imgs
from sklearn.model_selection import KFold
from data_loader import get_loader
from solver import Solver
from postprocessing import find_mean_std, getROC_curve, getPR_curve


def get_count(y):
	classes=["NMP","MP"]
	count=[0,0]
	for i in y:
		if i == "NMP": 
			count[0] += 1
		elif i == "MP": 
			count[1] += 1
	return count, classes


def main(config):
		
	cudnn.benchmark = True
	if config.model_type not in ['VGG16','ResNet18','SqueezeNet','MobileNetv2']:
		print('ERROR!! model_type should be selected in VGG16/ResNet18/SqueezeNet/MobileNetv2')
		print('Your input for model_type was %s'%config.model_type)
		return
	
	# Create new directories
	config = newdirs(config)
	
	# Initialize Reinhard stain normalization
	config = get_RNorm(config)
	
	# Preprocess cystectomy images (resize and stain normalize) and extract patches
	# Listing image and GT names
	images = [x for x in config.cysty_img_path.iterdir()]
	GT_imgs = [x for x in config.GT_cysty_path.iterdir()]
	config = get_patches(config, images, GT_imgs, config.train_patch_path_cyst, np.arange(len(images)), "cyst")
	config.cyst_mp = config.mp_patches; config.cyst_nmp = config.nmp_patches
	
	
	# Define k-fold cross validation parameters
	k=9; config.foldres = {}; patchinfo = {'fold': [], 'training_patches': [], 'mp_patches': [], 'nmp_patches': [], 'IoU': [], 'Dice Coefficient': [], 'Pixel-wise Accuracy': [], 'Precision': [], 'Recall': [], 'Specificity': [], 'F1 score': [], 'Optimal threshold': []}
	splits=KFold(n_splits=k,shuffle=True,random_state=42); max_iou = 0
	
	tur_images = [x for x in config.tur_img_path.iterdir()]
	tur_GT = [x for x in config.GT_tur_path.iterdir()]
	for fold, (config.train_idx, config.test_idx) in enumerate(splits.split(np.arange(len(tur_images)))):
		
		print("Fold {}".format(fold + 1)); patchinfo['fold'].append(fold+1)
		
		# Preprocess cystectomy images (resize and stain normalize) and extract patches
		config = get_patches(config, tur_images, tur_GT, config.train_patch_path_tur, config.train_idx, "tur")
		config.train_mp = config.mp_patches + config.cyst_mp; config.train_nmp = config.nmp_patches + config.cyst_nmp
		patchinfo['training_patches'].append(config.train_mp + config.train_nmp); patchinfo['mp_patches'].append(config.train_mp); patchinfo['nmp_patches'].append(config.train_nmp)
		
		# List training patches
		df_cys = pd.read_csv(config.train_patch_path.joinpath("cyst_patch_details_with_labels.csv"))
		df_tur = pd.read_csv(config.train_patch_path.joinpath("tur_patch_details_with_labels.csv"))
		cys_tur = [df_cys, df_tur]; df = pd.concat(cys_tur)
		X = df.iloc[:,0].to_list(); y = df.iloc[:,-1].to_list()
		
		# Divide training dataset into train set(80%) and validation set(20%)
		X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=69)
		
		# List test images
		X_test=[z.as_posix() for z in config.SN_test_path.iterdir()]
		
		# Determine class weights for training set
		count_train, config.classes_list = get_count(y_train)
		config.class_weights = 1./torch.tensor(count_train, dtype=torch.float)
		
		# Create data loaders for train and test set
		train_loader = get_loader(config.mean, config.std, config.batch_size, X_train, y_train, config, 'train')
		valid_loader = get_loader(config.mean, config.std, config.batch_size, X_val, y_val, config, 'validation')
		test_loader = get_loader(config.mean, config.std, 1, X_test, [0], config, 'test')
		
		# Initialize an object to train and test model
		solver = Solver(config, train_loader, valid_loader, test_loader, fold+1)
		
		print("Training initiated for Fold {}".format(fold+1))
		
		# Train the model
		solver.train()
		
		# Test the trained model
		solver.test()
		
		# Post-process predicted test images
		opt_thr, iou, dc, acc, pr, rc, sp, f1 = solver.post_processing()
		patchinfo['IoU'].append(iou); patchinfo['Dice Coefficient'].append(dc); patchinfo['Pixel-wise Accuracy'].append(acc)
		patchinfo['Precision'].append(pr); patchinfo['Recall'].append(rc); patchinfo['Specificity'].append(sp); patchinfo['F1 score'].append(f1); patchinfo['Optimal threshold'].append(opt_thr)
		print ("Fold {} completed\n\n".format(fold + 1))
		
		if iou >= max_iou:
			best_fold = fold+1
			max_iou = iou
			thr = opt_thr
			
	print("Best fold is " + str(best_fold) + " and best threshold is " + str(thr))		
	# List ambiguous images and marked images
	amb_imgs = [z for z in config.amb_image_path.iterdir()]
	amb_marked_imgs = [z for z in config.amb_mark_image_path.iterdir()]
	process_amb_imgs(config, amb_imgs, amb_marked_imgs)
	X_amb = [z.as_posix() for z in config.amb_SN_images.iterdir()]
	amb_loader = get_loader(config.mean, config.std, 1, X_amb, [0], config, 'test')
	solver.test_amb_imgs(best_fold, amb_loader, thr, config.amb_images)	
	df = pd.DataFrame(patchinfo)
	df.to_csv(Path(config.result_path).joinpath("Fold-wise_evaluation_metrics_infomation.csv"), index=False)
	
	eval_metric = ["IOU", "Dice Coefficient", "Pixel-wise Accuracy", "Precision", "Recall", "Specificity", "F1 score"]; mean_values = []; std_values = []
	iou_mean, iou_std, dc_mean, dc_std, acc_mean, acc_std, pr_mean, pr_std, rc_mean, rc_std, sp_mean, sp_std, f1_mean, f1_std = find_mean_std(patchinfo['IoU'], patchinfo['Dice Coefficient'], patchinfo['Pixel-wise Accuracy'], patchinfo['Precision'], patchinfo['Recall'], patchinfo['Specificity'], patchinfo['F1 score'])
	mean_values.append(iou_mean); mean_values.append(dc_mean); mean_values.append(acc_mean); mean_values.append(pr_mean); mean_values.append(rc_mean); mean_values.append(sp_mean); mean_values.append(f1_mean)
	std_values.append(iou_std); std_values.append(dc_std); std_values.append(acc_std); std_values.append(pr_std); std_values.append(rc_std); std_values.append(sp_std); std_values.append(f1_std)
	df = {'Eval metrics': eval_metric, 'Mean wrt 9-fold CV': mean_values, 'Standard deviation wrt 9-fold CV': std_values}
	df = pd.DataFrame(df)
	df.to_csv(Path(config.result_path).joinpath("Final_result.csv"), index=False)
	
	# Read csv files for plotting all fold PR and ROC curve
	csv_files = [f for f in config.y_info_path.iterdir()]
	getROC_curve(csv_files, config.result_path)
	getPR_curve(csv_files, config.result_path)
	print("All Done")



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# model
	parser.add_argument('--model_type', type=str, default='MobileNetv2', help='VGG16/ResNet18/SqueezeNet/MobileNetv2')
	
	# training hyper-parameters
	parser.add_argument('--patch_size', type=int, default=240)
	parser.add_argument('--classes', type=int, default=2)
	parser.add_argument('--num_epochs', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--mean', type=float, default=[0.485, 0.456, 0.406])
	parser.add_argument('--std', type=float, default=[0.229, 0.224, 0.225])
	
	# defining paths
	parser.add_argument('--cysty_img_path', type=str, default=Path('../../Data/Imgs_for_analysis/cases/cystectomy'))
	parser.add_argument('--GT_cysty_path', type=str, default=Path('../../Data/Imgs_for_analysis/masked_imgs/cystectomy'))
	parser.add_argument('--tur_img_path', type=str, default=Path('../../Data/Imgs_for_analysis/cases/tur'))
	parser.add_argument('--GT_tur_path', type=str, default=Path('../../Data/Imgs_for_analysis/masked_imgs/tur'))
	parser.add_argument('--mark_tur_path', type=str, default=Path('../../Data/Imgs_for_analysis/marked_imgs/tur'))
	parser.add_argument('--path', type=str, default=Path('../../Data/Patch_to_label'))
	parser.add_argument('--amb_image_path', type=str, default=Path('../../Data/Imgs_for_analysis/Ambiguous_images'))
	parser.add_argument('--amb_mark_image_path', type=str, default=Path('../../Data/Imgs_for_analysis/Ambiguous_marked_images'))
	
	
	# misc
	parser.add_argument('--SN_img_path', type=str, default=Path('../../Data/Imgs_for_analysis/cases/cystectomy/case_1_MP_000002.tif'))
	parser.add_argument('--cuda_idx', type=int, default=1)

	config = parser.parse_args()
	main(config)