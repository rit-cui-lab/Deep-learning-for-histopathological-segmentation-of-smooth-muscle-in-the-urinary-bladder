import argparse
import torch
from pathlib import Path
from sklearn.model_selection import KFold
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import cv2 as cv
import numpy as np
from preprocessing import newdirs, get_RNorm, get_patches, process_amb_imgs
import pandas as pd
from postprocessing import getROC_curve, getPR_curve, find_mean_std
from sklearn.model_selection import train_test_split


def get_plot(train_GTs):
    count=[0,0]
    classes=["NMP","MP"]
    for mask in train_GTs:
        mask_img = cv.imread(mask.as_posix(),0)
        mask_img = cv.resize(mask_img, (224,224), interpolation=cv.INTER_NEAREST)
        count[0] += np.sum(mask_img==0)
        count[1] += np.sum(mask_img==255)
    
    return count, classes


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['UNet','MAnet','DeepLabV3Plus','FPN']:
        print('ERROR!! model_type should be selected in UNet/MAnet/DeepLabV3Plus/FPN')
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
    config = get_patches(config, images, GT_imgs, config.train_patch_path_cyst, config.train_GT_path_cyst, np.arange(len(images)), "cyst")
    config.cyst_patches = config.patch_num

    
    # Define k-fold cross validation parameters
    k=9; config.foldres = {} 
    patchinfo_1 = {'Fold': [], 'IoU': [], 'Dice Coefficient': [], 'Pixel-wise Accuracy': [], 'Precision': [], 'Recall': [], 'Specificity': [], 'F1 score': [], 'Optimal threshold': []}
    patchinfo_2 = {'Fold': [], 'IoU': [], 'Dice Coefficient': [], 'Pixel-wise Accuracy': [], 'Precision': [], 'Recall': [], 'Specificity': [], 'F1 score': [], 'Optimal threshold': []}
    splits=KFold(n_splits=k,shuffle=True,random_state=42); p_max_iou = 0; w_max_iou = 0
    
    tur_images = [x for x in config.tur_img_path.iterdir()]
    tur_GT = [x for x in config.GT_tur_path.iterdir()]
    
    for fold, (config.train_idx, config.test_idx) in enumerate(splits.split(np.arange(len(tur_images)))):
      
      print("Fold {}".format(fold + 1)); patchinfo_1['Fold'].append(fold+1); patchinfo_2['Fold'].append(fold+1)
      
      # Preprocess cystectomy images (resize and stain normalize) and extract patches
      config = get_patches(config, tur_images, tur_GT, config.train_patch_path_tur, config.train_GT_path_tur, config.train_idx, "tur")        
      config.train_patches = config.patch_num + config.cyst_patches
      
      # List training patches
      X_cyst = [z for z in config.train_patch_path_cyst.iterdir()]; X_tur = [z for z in config.train_patch_path_tur.iterdir()]
      X = X_cyst + X_tur
      y_cyst = [z for z in config.train_GT_path_cyst.iterdir()]; y_tur = [z for z in config.train_GT_path_tur.iterdir()]
      y = y_cyst + y_tur
      
      # Divide training dataset into train set(80%) and validation set(20%)
      X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=69)
      
      # List test images
      X_test=[z for z in config.SN_test_path.iterdir()]
      
      # Determine class weights for training set
      count_train, classes_train = get_plot(y_train)
      config.class_weights = 1./torch.tensor(count_train, dtype=torch.float)
      
      # Create data loaders for train and test set
      train_loader = get_loader(config.mean, config.std, config.batch_size, X_train, y_train, config, 'train')
      valid_loader = get_loader(config.mean, config.std, config.batch_size, X_val, y_val, config, 'valid')
      test_loader = get_loader(config.mean, config.std, 1, X_test, [0], config, 'test')
      
      # Initialize an object to train and test model
      solver = Solver(config, train_loader, valid_loader, test_loader, fold+1)
      
      print("Training initiated for Fold {}".format(fold+1))
      
      # Train the model
      solver.train()
      
      # Test the trained model using patch-based approach
      solver.patch_based_seg()
      
      # Test the trained model using whole image-based approach
      solver.whole_img_based_seg()
      
      # Post-process predicted test images
      p_opt_thr, p_iou, p_dc, p_acc, p_pr, p_rc, p_sp, p_f1, w_opt_thr, w_iou, w_dc, w_acc, w_pr, w_rc, w_sp, w_f1 = solver.post_processing()
      patchinfo_1['IoU'].append(p_iou); patchinfo_1['Dice Coefficient'].append(p_dc); patchinfo_1['Pixel-wise Accuracy'].append(p_acc); patchinfo_1['Precision'].append(p_pr); patchinfo_1['Recall'].append(p_rc); patchinfo_1['Specificity'].append(p_sp); patchinfo_1['F1 score'].append(p_f1); patchinfo_1['Optimal threshold'].append(p_opt_thr)
      patchinfo_2['IoU'].append(w_iou); patchinfo_2['Dice Coefficient'].append(w_dc); patchinfo_2['Pixel-wise Accuracy'].append(w_acc); patchinfo_2['Precision'].append(w_pr); patchinfo_2['Recall'].append(w_rc); patchinfo_2['Specificity'].append(w_sp); patchinfo_2['F1 score'].append(w_f1); patchinfo_2['Optimal threshold'].append(w_opt_thr)      
      print ("Fold {} completed\n\n".format(fold + 1))
      
      if p_iou >= p_max_iou:
          p_best_fold = fold+1
          p_max_iou = p_iou
          p_thr = p_opt_thr
          
      if w_iou >= w_max_iou:
          w_best_fold = fold+1
          w_max_iou = w_iou
          w_thr = w_opt_thr
           
    print("Best fold for patch-based inference is " + str(p_best_fold) + " and best threshold is " + str(p_thr))
    print("Best fold for whole image-based inference is " + str(w_best_fold) + " and best threshold is " + str(w_thr))        
    
    df = pd.DataFrame(patchinfo_1)
    df.to_csv(Path(config.result_path).joinpath("Fold-wise_evaluation_metrics_infomation_patch_based_approach.csv"), index=False)
    df = pd.DataFrame(patchinfo_2)
    df.to_csv(Path(config.result_path).joinpath("Fold-wise_evaluation_metrics_infomation_whole_image_based_approach.csv"), index=False)
    
    # List ambiguous images and marked images
    amb_imgs = [z for z in config.amb_image_path.iterdir()]
    amb_marked_imgs = [z for z in config.amb_mark_image_path.iterdir()]
    process_amb_imgs(config, amb_imgs, amb_marked_imgs)
    X_amb = [z for z in config.amb_SN_images.iterdir()]
    amb_loader = get_loader(config.mean, config.std, 1, X_amb, [0], config, 'test')
    
    # Test ambiguous images using patch-based inference method
    solver.test_amb_imgs(p_best_fold, amb_loader, p_thr, config.amb_images, "patch_based")  

    # Test ambiguous images using patch-based inference method
    solver.test_amb_imgs(w_best_fold, amb_loader, w_thr, config.amb_images, "wholeIMG_based")   
        
    # Patch-based approach evaluation
    eval_metric = ["IOU", "Dice Coefficient", "Pixel-wise Accuracy", "Precision", "Recall", "Specificity", "F1 score"]; mean_values = []; std_values = []
    iou_mean, iou_std, dc_mean, dc_std, acc_mean, acc_std, pr_mean, pr_std, rc_mean, rc_std, sp_mean, sp_std, f1_mean, f1_std = find_mean_std(patchinfo_1['IoU'], patchinfo_1['Dice Coefficient'], patchinfo_1['Pixel-wise Accuracy'], patchinfo_1['Precision'], patchinfo_1['Recall'], patchinfo_1['Specificity'], patchinfo_1['F1 score'])    
    mean_values.append(iou_mean); mean_values.append(dc_mean); mean_values.append(acc_mean); mean_values.append(pr_mean); mean_values.append(rc_mean); mean_values.append(sp_mean); mean_values.append(f1_mean)
    std_values.append(iou_std); std_values.append(dc_std); std_values.append(acc_std); std_values.append(pr_std); std_values.append(rc_std); std_values.append(sp_std); std_values.append(f1_std)
    df = {'Eval metrics': eval_metric, 'Mean wrt 9-fold CV': mean_values, 'Standard deviation wrt 9-fold CV': std_values}
    df = pd.DataFrame(df)
    df.to_csv(Path(config.result_path).joinpath("Patch_based_approach_result.csv"), index=False)
    # Read csv files for plotting all fold PR and ROC curve for patch-based predicted images
    csv_files_mean = [f for f in config.y_info_path_mean.iterdir()]
    getROC_curve(csv_files_mean, config.result_path, "patch_based")
    getPR_curve(csv_files_mean, config.result_path, "patch_based")

    
    # Whole-image based approach evaluation
    eval_metric = ["IOU", "Dice Coefficient", "Pixel-wise Accuracy", "Precision", "Recall", "Specificity", "F1 score"]; mean_values = []; std_values = []
    iou_mean, iou_std, dc_mean, dc_std, acc_mean, acc_std, pr_mean, pr_std, rc_mean, rc_std, sp_mean, sp_std, f1_mean, f1_std = find_mean_std(patchinfo_2['IoU'], patchinfo_2['Dice Coefficient'], patchinfo_2['Pixel-wise Accuracy'], patchinfo_2['Precision'], patchinfo_2['Recall'], patchinfo_2['Specificity'], patchinfo_2['F1 score'])
    mean_values.append(iou_mean); mean_values.append(dc_mean); mean_values.append(acc_mean); mean_values.append(pr_mean); mean_values.append(rc_mean); mean_values.append(sp_mean); mean_values.append(f1_mean)
    std_values.append(iou_std); std_values.append(dc_std); std_values.append(acc_std); std_values.append(pr_std); std_values.append(rc_std); std_values.append(sp_std); std_values.append(f1_std)
    df = {'Eval metrics': eval_metric, 'Mean wrt 9-fold CV': mean_values, 'Standard deviation wrt 9-fold CV': std_values}
    df = pd.DataFrame(df)
    df.to_csv(Path(config.result_path).joinpath("Whole_image_based_approach_result.csv"), index=False)
    # Read csv files for plotting all fold PR and ROC curve for whole image-based predicted images
    csv_files_full_img = [f for f in config.y_info_path_fullimg.iterdir()]
    getROC_curve(csv_files_full_img, config.result_path, "wholeIMG_based")
    getPR_curve(csv_files_full_img, config.result_path, "wholeIMG_based")
    print("All Done")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_type', type=str, default='FPN', help='UNet/MAnet/DeepLabV3Plus/FPN')
    
    # training hyper-parameters
    parser.add_argument('--patch_size', type=int, default=240)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)
    parser.add_argument('--mean', type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument('--std', type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument('--mapping', type=float, default={0: 0, 255: 1})
    
    # defining paths
    parser.add_argument('--cysty_img_path', type=str, default=Path('../../Data/Imgs_for_analysis/cases/cystectomy'))
    parser.add_argument('--GT_cysty_path', type=str, default=Path('../../Data/Imgs_for_analysis/masked_imgs/cystectomy'))
    parser.add_argument('--tur_img_path', type=str, default=Path('../../Data/Imgs_for_analysis/cases/tur'))
    parser.add_argument('--GT_tur_path', type=str, default=Path('../../Data/Imgs_for_analysis/masked_imgs/tur'))
    parser.add_argument('--mark_tur_path', type=str, default=Path('../../Data/Imgs_for_analysis/marked_imgs/tur'))
    parser.add_argument('--path', type=str, default=Path('../../Data/Pixel_to_label'))
    parser.add_argument('--amb_image_path', type=str, default=Path('../../Data/Imgs_for_analysis/Ambiguous_images'))
    parser.add_argument('--amb_mark_image_path', type=str, default=Path('../../Data/Imgs_for_analysis/Ambiguous_marked_images'))
    
    # misc
    parser.add_argument('--SN_img_path', type=str, default=Path('../../Data/Imgs_for_analysis/cases/cystectomy/case_1_MP_000002.tif'))
    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)