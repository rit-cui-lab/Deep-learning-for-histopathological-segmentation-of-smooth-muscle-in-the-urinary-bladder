import numpy as np
import cv2 as cv
import pandas as pd
from pathlib import Path
from sklearn.metrics import auc, precision_recall_curve, roc_curve, average_precision_score
import matplotlib.pyplot as plt


def precision_recall_curve_plot(y_true, prob_MP, path, fold, mode):
    precision, recall, thr = precision_recall_curve(y_true, prob_MP)
    fscore = (2*precision*recall)/(precision+recall)
    ix = np.argmax(fscore)
    PR_optimal_thr = thr[ix]
    print ("PR Optimal threshold is: "+str(PR_optimal_thr))
    auc_precision = auc(recall,precision)
    no_skill = y_true.count(1)/len(y_true)
    plt.figure()
    plt.title('Precision Recall Curve (PR)')
    plt.plot(recall,precision, 'b', marker='.', label = 'Average PR score = %0.3f' %auc_precision)
    plt.plot([0,1], [no_skill, no_skill], 'r--', label='No Skill')
    plt.legend()
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.1])
    plt.ylabel('Precision')
    plt.xlabel('Recall (Sensitivity)')
    plt.show()
    plt.savefig(Path(path).joinpath("Pixelwise_PR_AUC_curve_fold_"+fold+"_"+mode+".png"))
    return PR_optimal_thr


def ROC_plot(y_true, prob_MP, path, fold, mode):
    fpr, tpr, thresholds = roc_curve(y_true, prob_MP)
    roc_auc_score=auc(fpr,tpr)
    optimal_idx = np.argmax(tpr - fpr)
    ROC_optimal_threshold = thresholds[optimal_idx]
    print ("ROC Optimal threshold is: "+str(ROC_optimal_threshold))
    plt.figure()
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.plot(fpr,tpr, 'b', label = 'AUC = %0.3f' %roc_auc_score)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([0.0,1.1])
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate (1-Specificity)')
    plt.show()
    plt.savefig(Path(path).joinpath("Pixelwise_ROC_AUC_curve_fold_"+fold+"_"+mode+".png"))
    return ROC_optimal_threshold


def get_thr(pred_images, GT_images, res_path, fold, y_info_path, mode):
    y_true=np.array([]);prob_MP=np.array([])
    for i in range(len(pred_images)):
        if(GT_images[i].parts[-1].split('_')[0:4]==(pred_images[i].parts[-1].split('_')[0:4])):
            img=cv.imread(pred_images[i].as_posix(),0)/255
            GT=cv.imread(GT_images[i].as_posix(),0)
            GT=GT/255
            
            y_true=np.append(y_true, GT.flatten())
            prob_MP=np.append(prob_MP, img.flatten())

        else:
            print ("Name mismatch: "+pred_images[i].parts[-1]+" "+GT_images[i].parts[-1])
    
    y_true=list(y_true)
    prob_MP=list(prob_MP)
    df={"y_true": y_true, "prob_MP": prob_MP}
    df=pd.DataFrame(df)
    df.to_csv(Path(y_info_path).joinpath("Ytrue_Yprob_pixelwise_fold_"+fold+"_"+mode+".csv"), index=False)
    
    pr_thr=precision_recall_curve_plot(y_true, prob_MP, res_path, fold, mode)
    roc_thr=ROC_plot(y_true, prob_MP, res_path, fold, mode)
    return pr_thr, roc_thr


def get_tp_fp_fn(pred, GT, clss, threshold=0.5):
    pred = pred > threshold
    GT = GT == 1
    if (clss == "non-MP"):
        pred = np.logical_not(pred)
        GT = np.logical_not(GT)
    TP = np.sum(np.logical_and(pred==True, GT==True))
    TN = np.sum(np.logical_and(pred==False, GT==False))
    FP = np.sum(np.logical_and(pred==True, GT==False))
    FN = np.sum(np.logical_and(pred==False, GT==True))
    return TP, TN, FP, FN 


def find_metric(mp_tp, mp_tn, mp_fp, mp_fn, nmp_tp, nmp_tn, nmp_fp, nmp_fn):
    if mp_tp != 0 or mp_fp != 0 or mp_fn != 0:
        mp_iou = float(mp_tp)/float(mp_tp + mp_fp + mp_fn)
        mp_DC = float(2 * mp_iou)/float(1 + mp_iou)
    else:
        mp_iou = 0
        mp_DC = 0
    
    if nmp_tp != 0 or nmp_fp != 0 or nmp_fn != 0:
        nmp_iou = float(nmp_tp)/float(nmp_tp + nmp_fp + nmp_fn)
        nmp_DC = float(2 * nmp_iou)/float(1 + nmp_iou)
    else:
        nmp_iou = 0
        nmp_DC = 0
    
    if mp_iou == 0 or nmp_iou == 0:
        avg_iou = mp_iou + nmp_iou
        avg_DC = mp_DC + nmp_DC
        
    else:
        avg_iou = (mp_iou + nmp_iou)/2
        avg_DC = (mp_DC + nmp_DC)/2
    
    return avg_iou, avg_DC


def find_all_metric(mp_tp, mp_tn, mp_fp, mp_fn, nmp_tp, nmp_tn, nmp_fp, nmp_fn):
    mp_iou = float(mp_tp)/float(mp_tp + mp_fp + mp_fn)
    nmp_iou = float(nmp_tp)/float(nmp_tp + nmp_fp + nmp_fn)
    avg_iou = (mp_iou + nmp_iou)/2
    
    mp_DC = float(2 * mp_iou)/float(1 + mp_iou)
    nmp_DC = float(2 * nmp_iou)/float(1 + nmp_iou)
    avg_DC = (mp_DC + nmp_DC)/2
		
    pixel_acc = float(mp_tp + mp_tn)/float(mp_tp + mp_tn + mp_fp + mp_fn)

    PC = float(mp_tp)/float(mp_tp + mp_fp)    

    recall = float(mp_tp)/float(mp_tp + mp_fn)
		
    SP = float(mp_tn)/float(mp_tn + mp_fp)
		
    F1 = (2 * PC * recall)/(PC + recall)
    
    return avg_iou, avg_DC, pixel_acc, PC, recall, SP, F1


def get_eval_metrics(pred_images, GT_images, res_path, fold, pr_thr, roc_thr, mode):
    eval_dict = {"Image_name": [], "MP_true_positive": [], "MP_true_negative": [], "MP_false_positive": [], "MP_false_negative": [], 
                 "NMP_true_positive": [], "NMP_true_negative": [], "NMP_false_positive": [], "NMP_false_negative": [], 
                 "Mean-Jaccard_index": [], "Dice_coefficient": []}
    mp_tp = 0; mp_tn = 0; mp_fp = 0; mp_fn = 0; nmp_tp = 0; nmp_tn = 0; nmp_fp = 0; nmp_fn = 0
    all_mp_tp = 0; all_mp_tn = 0; all_mp_fp = 0; all_mp_fn = 0; all_nmp_tp = 0; all_nmp_tn = 0; all_nmp_fp = 0; all_nmp_fn = 0
    for i in range(len(pred_images)):
        if(GT_images[i].parts[-1].split('_')[0:4]==(pred_images[i].parts[-1].split('_')[0:4])):
            eval_dict["Image_name"].append(pred_images[i].parts[-1])
            img=cv.imread(pred_images[i].as_posix(),0)/255
            GT=cv.imread(GT_images[i].as_posix(),0)
            GT=GT/255
            mp_tp, mp_tn, mp_fp, mp_fn = get_tp_fp_fn(img, GT, "MP")
            all_mp_tp += mp_tp; all_mp_tn += mp_tn; all_mp_fp += mp_fp; all_mp_fn += mp_fn
            eval_dict["MP_true_positive"].append(mp_tp); eval_dict["MP_true_negative"].append(mp_tn); eval_dict["MP_false_positive"].append(mp_fp); eval_dict["MP_false_negative"].append(mp_fn)
            
            nmp_tp, nmp_tn, nmp_fp, nmp_fn = get_tp_fp_fn(img, GT, "non-MP")
            all_nmp_tp += nmp_tp; all_nmp_tn += nmp_tn; all_nmp_fp += nmp_fp; all_nmp_fn += nmp_fn
            eval_dict["NMP_true_positive"].append(nmp_tp); eval_dict["NMP_true_negative"].append(nmp_tn); eval_dict["NMP_false_positive"].append(nmp_fp); eval_dict["NMP_false_negative"].append(nmp_fn)     
            
            avg_iou, avg_DC = find_metric(mp_tp, mp_tn, mp_fp, mp_fn, nmp_tp, nmp_tn, nmp_fp, nmp_fn)
            eval_dict["Mean-Jaccard_index"].append(avg_iou); eval_dict["Dice_coefficient"].append(avg_DC)
        else:
            print ("Name mismatch: "+pred_images[i].parts[-1]+" "+GT_images[i].parts[-1])
    
    avg_iou, avg_DC, pixel_acc, PC, recall, SP, F1 = find_all_metric(all_mp_tp, all_mp_tn, all_mp_fp, all_mp_fn, all_nmp_tp, all_nmp_tn, all_nmp_fp, all_nmp_fn)
    file = open(Path(res_path).joinpath("Fold_"+fold+"_"+mode+"_evaluation_metrics.txt"), "w")
    file.write("Evaluation metrics considering all test images in this fold")
    file.write("\nAverage Jaccard Index: " + str(avg_iou))
    file.write("\nAverage Dice coefficient: " + str(avg_DC))
    file.write("\nPixel-wise accuracy: " + str(pixel_acc))
    file.write("\nPrecision: " + str(PC))
    file.write("\nRecall/Sensitivity: " + str(recall))
    file.write("\nSpecificity: " + str(SP))
    file.write("\nF1-score: " + str(F1))
    file.write("\nPR optimal threshold: " + str(pr_thr))
    file.write("\nROC optimal threshold: " + str(roc_thr))
    file.close()
    
    df = pd.DataFrame(eval_dict)
    df.to_csv(Path(res_path).joinpath("Pixelwise_Model_evaluation_result_fold_"+fold+"_"+mode+".csv"), index=False)
    return avg_iou, avg_DC, pixel_acc, PC, recall, SP, F1


def get_postprocessed_imgs(pred_imgs, imgs, HM_path, contour_path, opt_thr):
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
				cv.imwrite(HM_path.joinpath(img_fname).as_posix(),output_img)
				
				# find the contours from the thresholded image
				contours, hierarchy = cv.findContours(output_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
				
				# draw all contours
				output_img = cv.drawContours(org_img, contours, -1, (0, 0, 255), 10)
				
				# Save the final_img
				cv.imwrite(contour_path.joinpath(img_fname).as_posix(),output_img)



def getROC_curve(files, path, mode):
    plt.figure(); tprs = []; aucs = []; mean_fpr = np.linspace(0, 1, 1920*1440*7)
    
    for i in range(len(files)):
        
        # Read csv file
        df = pd.read_csv(files[i])
        
        # Find trp and fpr (ROC curve) for all folds
        fpr, tpr, thr = roc_curve(df['y_true'], df['prob_MP'])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr); aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    # plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(Path(path).joinpath("9-fold Cross Validation ROC curve_"+mode+".png"), bbox_inches='tight', dpi=300)
    
    
def getPR_curve(files, path, mode):
    
    plt.figure(); prs = []; aucs = []; mean_recall = np.linspace(0, 1, 1920*1440*7)
    
    for i in range(len(files)):
        
        # Read csv file
        df = pd.read_csv(files[i])
        
        # Find precision and recall (PR curve) for each fold and plot fold PR curve
        p, r, _ = precision_recall_curve(df['y_true'], df['prob_MP'])
        prs.append(np.interp(mean_recall, p, r))
        pr_auc = auc(r, p)
        aucs.append(pr_auc)
        plt.plot(r, p, lw=1, alpha=0.3, label='PR fold %d (AUC = %0.2f)' % (i+1, pr_auc))
    
    plt.plot([0, 1], [1, 0], linestyle='--', lw=3, color='k', label='Luck', alpha=.8)
    mean_precision = np.mean(prs, axis=0)
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(aucs)
    plt.plot(mean_recall, mean_precision, color='b', label=r'Mean (PR AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    
    std_prs = np.std(prs, axis=0)
    prs_upper = np.minimum(mean_precision + std_prs, 1)
    prs_lower = np.maximum(mean_precision - std_prs, 0)
    plt.fill_between(mean_recall, prs_lower, prs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    # plt.title("Precision Recall Curve")
    plt.legend(loc="lower left")
    plt.show()
    plt.savefig(Path(path).joinpath("9-fold Cross Validation PR curve_"+mode+".png"), bbox_inches='tight', dpi=300)

def find_mean_std(iou, dc, acc, pr, rc, sp, f1):
    iou_mean = np.mean(iou); iou_std = np.std(iou)
    dc_mean = np.mean(dc); dc_std = np.std(dc)
    acc_mean = np.mean(acc); acc_std = np.std(acc)
    pr_mean = np.mean(pr); pr_std = np.std(pr)
    rc_mean = np.mean(rc); rc_std = np.std(rc)
    sp_mean = np.mean(sp); sp_std = np.std(sp)
    f1_mean = np.mean(f1); f1_std = np.std(f1)
    return iou_mean, iou_std, dc_mean, dc_std, acc_mean, acc_std, pr_mean, pr_std, rc_mean, rc_std, sp_mean, sp_std, f1_mean, f1_std
    
    