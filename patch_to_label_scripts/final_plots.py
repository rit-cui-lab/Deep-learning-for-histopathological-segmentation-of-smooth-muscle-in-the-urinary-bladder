import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def boxplot(config, data, mode):
    fig_name = mode + "_box_plot.png"
    plt.figure()
    meanlineprops = dict(linestyle='-', linewidth=1.5, color='blue')
    bp=plt.boxplot(data, labels=config.models, widths=0.4, showmeans=True, meanline=True, meanprops=meanlineprops)
    plt.setp(bp["medians"], color='black')
    plt.xlabel('Patch-based Models', fontsize=15)
    plt.ylabel(mode, fontsize=15)
    plt.xticks(np.arange(1,5), config.models, rotation=0, fontsize=12)
    plt.yticks(np.arange(0.5,1.1,0.1)[0:-1], fontsize=12)
    # plt.title("Box-plot of "+mode, fontsize=15)
    plt.legend([bp['means'][0]], ['mean'], loc='lower right', fontsize=12)
    plt.show()
    plt.savefig(Path(config.path).joinpath(fig_name), bbox_inches='tight', dpi=300)
    plt.close()


def main(config):
    
    fname = "Fold-wise_evaluation_metrics_infomation.csv"

    # Read the csv files
    vgg_df = pd.read_csv(config.vgg_path.joinpath(fname))
    resnet_df = pd.read_csv(config.resnet_path.joinpath(fname))
    squeezenet_df = pd.read_csv(config.squeezenet_path.joinpath(fname))
    mobilenet_df = pd.read_csv(config.mobilenet_path.joinpath(fname))
    
    config.fold = vgg_df.iloc[:,0].to_list()
    
    # Iou - 4th column; DC - 5th column; acc - 6th column; precision - 7th column; recall - 8th column; specificity - 9th column; F1 - 10th column
    iou_data = []; dc_data = []; acc_data = []; pc_data = []; rc_data = []; sp_data = []; f1_data = []
    iou_data.append(vgg_df.iloc[:,4].to_list()); iou_data.append(resnet_df.iloc[:,4].to_list()); iou_data.append(squeezenet_df.iloc[:,4].to_list()); iou_data.append(mobilenet_df.iloc[:,4].to_list())
    dc_data.append(vgg_df.iloc[:,5].to_list()); dc_data.append(resnet_df.iloc[:,5].to_list()); dc_data.append(squeezenet_df.iloc[:,5].to_list()); dc_data.append(mobilenet_df.iloc[:,5].to_list())
    acc_data.append(vgg_df.iloc[:,6].to_list()); acc_data.append(resnet_df.iloc[:,6].to_list()); acc_data.append(squeezenet_df.iloc[:,6].to_list()); acc_data.append(mobilenet_df.iloc[:,6].to_list())
    pc_data.append(vgg_df.iloc[:,7].to_list()); pc_data.append(resnet_df.iloc[:,7].to_list()); pc_data.append(squeezenet_df.iloc[:,7].to_list()); pc_data.append(mobilenet_df.iloc[:,7].to_list())
    rc_data.append(vgg_df.iloc[:,8].to_list()); rc_data.append(resnet_df.iloc[:,8].to_list()); rc_data.append(squeezenet_df.iloc[:,8].to_list()); rc_data.append(mobilenet_df.iloc[:,8].to_list())
    sp_data.append(vgg_df.iloc[:,9].to_list()); sp_data.append(resnet_df.iloc[:,9].to_list()); sp_data.append(squeezenet_df.iloc[:,9].to_list()); sp_data.append(mobilenet_df.iloc[:,9].to_list())
    f1_data.append(vgg_df.iloc[:,10].to_list()); f1_data.append(resnet_df.iloc[:,10].to_list()); f1_data.append(squeezenet_df.iloc[:,10].to_list()); f1_data.append(mobilenet_df.iloc[:,10].to_list())
    
    config.models = ["VGG16", "ResNet18", "SqueezeNet", "MobileNet"]     
    
    # Build the box-plots for each evaluation metrics
    boxplot(config, iou_data, "Mean Jaccard Index")
    boxplot(config, dc_data, "Mean Dice Coefficient")
    boxplot(config, acc_data, "Pixelwise Accuracy")            
    boxplot(config, pc_data, "Precision")
    boxplot(config, rc_data, "Recall")
    boxplot(config, sp_data, "Specificity")
    boxplot(config, f1_data, "F1 score")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model result path
    parser.add_argument('--path', type=str, default=Path('../../Data/Patch_to_label'))
    parser.add_argument('--vgg_path', type=str, default=Path('../../Data/Patch_to_label/VGG16/results'))
    parser.add_argument('--resnet_path', type=str, default=Path('../../Data/Patch_to_label/ResNet18/results'))
    parser.add_argument('--squeezenet_path', type=str, default=Path('../../Data/Patch_to_label/SqueezeNet/results'))
    parser.add_argument('--mobilenet_path', type=str, default=Path('../../Data/Patch_to_label/MobileNetv2/results'))
    
    config = parser.parse_args()
    main(config)