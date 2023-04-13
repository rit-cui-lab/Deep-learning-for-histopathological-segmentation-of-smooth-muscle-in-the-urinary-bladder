import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def boxplot(config, data, approach, mode):
    fig_name = approach + mode + "_box_plot.png"
    plt.figure()
    meanlineprops = dict(linestyle='-', linewidth=1.5, color='blue')
    bp=plt.boxplot(data, labels=config.models, widths=0.4, showmeans=True, meanline=True, meanprops=meanlineprops)
    plt.setp(bp["medians"], color='black')
    plt.xlabel('Pixel-based Models', fontsize=15)
    plt.ylabel(mode, fontsize=15)
    plt.xticks(np.arange(1,5), config.models, rotation=0, fontsize=12)
    plt.yticks(np.arange(0.5,1.1,0.1)[0:-1], fontsize=12)
    # plt.title("Box-plot of "+mode, fontsize=15)
    plt.legend([bp['means'][0]], ['mean'], loc='lower right', fontsize=12)
    plt.show()
    plt.savefig(Path(config.path).joinpath(fig_name), bbox_inches='tight', dpi=300)
    plt.close()


def main(config):
    
    # fname = "Fold-wise_evaluation_metrics_infomation_patch_based_approach.csv"; approach = "Patch_based_"
    fname = "Fold-wise_evaluation_metrics_infomation_whole_image_based_approach.csv"; approach = "Whole_Img_based_"

    # Read tthe csv files
    unet_df = pd.read_csv(config.unet_path.joinpath(fname))
    manet_df = pd.read_csv(config.manet_path.joinpath(fname))
    deeplabv3_df = pd.read_csv(config.deeplabv3_path.joinpath(fname))
    fpn_df = pd.read_csv(config.fpn_path.joinpath(fname))
    
    config.fold = unet_df.iloc[:,0].to_list()
    
    # Iou - 1st column; DC - 2nd column; acc - 3rd column; precision - 4th column; recall - 5th column; specificity - 6th column; F1 - 7th column
    iou_data = []; dc_data = []; acc_data = []; pc_data = []; rc_data = []; sp_data = []; f1_data = []
    iou_data.append(unet_df.iloc[:,1].to_list()); iou_data.append(manet_df.iloc[:,1].to_list()); iou_data.append(deeplabv3_df.iloc[:,1].to_list()); iou_data.append(fpn_df.iloc[:,1].to_list())
    dc_data.append(unet_df.iloc[:,2].to_list()); dc_data.append(manet_df.iloc[:,2].to_list()); dc_data.append(deeplabv3_df.iloc[:,2].to_list()); dc_data.append(fpn_df.iloc[:,2].to_list())
    acc_data.append(unet_df.iloc[:,3].to_list()); acc_data.append(manet_df.iloc[:,3].to_list()); acc_data.append(deeplabv3_df.iloc[:,3].to_list()); acc_data.append(fpn_df.iloc[:,3].to_list())
    pc_data.append(unet_df.iloc[:,4].to_list()); pc_data.append(manet_df.iloc[:,4].to_list()); pc_data.append(deeplabv3_df.iloc[:,4].to_list()); pc_data.append(fpn_df.iloc[:,4].to_list())
    rc_data.append(unet_df.iloc[:,5].to_list()); rc_data.append(manet_df.iloc[:,5].to_list()); rc_data.append(deeplabv3_df.iloc[:,5].to_list()); rc_data.append(fpn_df.iloc[:,5].to_list())
    sp_data.append(unet_df.iloc[:,6].to_list()); sp_data.append(manet_df.iloc[:,6].to_list()); sp_data.append(deeplabv3_df.iloc[:,6].to_list()); sp_data.append(fpn_df.iloc[:,6].to_list())
    f1_data.append(unet_df.iloc[:,7].to_list()); f1_data.append(manet_df.iloc[:,7].to_list()); f1_data.append(deeplabv3_df.iloc[:,7].to_list()); f1_data.append(fpn_df.iloc[:,7].to_list())
    
    config.models = ["U-Net", "MA-Net", "DeepLabv3+", "FPN"]     
    
    
    # Build the box-plots for each evaluation metrics
    boxplot(config, iou_data, approach, "Mean Jaccard Index")
    boxplot(config, dc_data, approach, "Mean Dice Coefficient")
    boxplot(config, acc_data, approach, "Pixelwise Accuracy")            
    boxplot(config, pc_data, approach, "Precision")
    boxplot(config, rc_data, approach, "Recall")
    boxplot(config, sp_data, approach, "Specificity")
    boxplot(config, f1_data, approach, "F1 score")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model result path
    parser.add_argument('--path', type=str, default=Path('../../Data/Pixel_to_label'))
    parser.add_argument('--unet_path', type=str, default=Path('../../Data/Pixel_to_label/UNet/results'))
    parser.add_argument('--manet_path', type=str, default=Path('../../Data/Pixel_to_label/MAnet/results'))
    parser.add_argument('--deeplabv3_path', type=str, default=Path('../../Data/Pixel_to_label/DeepLabV3Plus/results'))
    parser.add_argument('--fpn_path', type=str, default=Path('../../Data/Pixel_to_label/FPN/results'))
    
    config = parser.parse_args()
    main(config)