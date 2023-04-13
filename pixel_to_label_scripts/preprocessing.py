from pathlib import Path
import shutil
import cv2 as cv
import stainNorm_Reinhard



def newdirs(config):
    config.amb_images = create_dir("amb_imgs", config.path, del_val = "False")
    config.amb_SN_images = create_dir("amb_SN_imgs", config.path, del_val = "False")
    config.amb_mark_images = create_dir("amb_mark_imgs", config.path, del_val = "False")
    config.path = create_dir(config.model_type, config.path, del_val = "False")
    config.model_path = create_dir("models", config.path, del_val = "False")
    config.result_path = create_dir("results", config.path, del_val = "False")
    config.train_patch_path = create_dir("training_patches", config.path, del_val = "False")
    config.train_patch_path_cyst = create_dir("cystectomy_patches", config.train_patch_path, del_val = "False")
    config.train_GT_path_cyst = create_dir("cystectomy_GT", config.train_patch_path, del_val = "False")
    config.train_patch_path_tur = create_dir("tur_patches", config.train_patch_path, del_val = "False")
    config.train_GT_path_tur = create_dir("tur_GT", config.train_patch_path, del_val = "False")
    config.test_img_path = create_dir("test_imgs", config.path, del_val = "False")
    config.SN_test_path = create_dir("SN_test_imgs", config.path, del_val = "False")
    config.GT_test_path = create_dir("test_GT", config.path, del_val = "False")
    config.marked_test_path = create_dir("test_marked", config.path, del_val = "False")
    config.y_info_path_mean = create_dir("y_info_patch_based", config.result_path, del_val = "False")
    config.y_info_path_fullimg = create_dir("y_info_wholeIMG_based", config.result_path, del_val = "False")
    
    return config


def create_dir(model, parent, del_val):
    if model == "null":
        source = parent
    else:
        source = parent.joinpath(model)
        
    if source.is_dir() and del_val == "True":
        shutil.rmtree(source)
        source.mkdir()
        print("New directory " + str(Path(source)) + " is created")
    elif not source.is_dir():
        source.mkdir()
        print("New directory " + str(Path(source)) + " is created")
    else:
        print ("Folder "+ str(Path(source)) + " exists")
    return source
    

def get_RNorm(config):
    target=cv.imread(config.SN_img_path.as_posix(),1) # 1 for color image and 0 for grey image
    target=cv.cvtColor(target,cv.COLOR_BGR2RGB)
    Norm=stainNorm_Reinhard.Normalizer()
    Norm.fit(target)
    config.Norm = Norm
    return config


def get_optimal_size(img_list, GT):
    tmp_list = []
    for img in img_list:
        img = cv.resize(img, (1920,1440), interpolation=cv.INTER_LINEAR) 
        tmp_list.append(img)
    GT = cv.resize(GT, (1920,1440), interpolation=cv.INTER_NEAREST)  
    return tmp_list, GT


def get_stainNorm(config, img):
    img = config.Norm.transform(img)
    return img 
   
 
def get_patchNum(num):
    return str(num).zfill(6)


def get_patches(config, images, GT_imgs, patch_path, mask_path, train_idx, img_type):
    
    # Initialize variables to store patch information
    config.patch_num=0
    
    # Create new directories if the img_type = "tur"
    if img_type == "tur":
        create_dir("null", patch_path, del_val = "True")
        create_dir("null", mask_path, del_val = "True")
        create_dir("null", config.SN_test_path, del_val = "True")
        create_dir("null", config.test_img_path, del_val = "True")
        create_dir("null", config.GT_test_path, del_val = "True")
        mark_imgs = [x for x in config.mark_tur_path.iterdir()]
        
        # Copy pre-processed test TUR images and their GT to test folder
        for i in range(len(config.test_idx)):
            img_fname = str(images[config.test_idx[i]]).split('\\')[-1]
            img = cv.imread(images[config.test_idx[i]].as_posix())
            SN_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            
            GT_fname = str(GT_imgs[config.test_idx[i]]).split('\\')[-1]
            GT = cv.imread(GT_imgs[config.test_idx[i]].as_posix())
            
            mark_fname = mark_imgs[config.test_idx[i]].parts[-1]
            mark_img = cv.imread(mark_imgs[config.test_idx[i]].as_posix())
            
            # Pre-process the test images
            if (img.shape != (1440, 1920, 3)):
                tmp_list, GT = get_optimal_size([img, SN_img, mark_img], GT)
                img = tmp_list[0]; SN_img = tmp_list[1]; mark_img = tmp_list[2]
            SN_img = get_stainNorm(config, SN_img)
            SN_img = cv.cvtColor(SN_img,cv.COLOR_RGB2BGR)
            GT[GT == 128] = 0
            
            # Save the pre-processed test images and their GTs to designated folders
            cv.imwrite(Path(config.test_img_path).joinpath(img_fname).as_posix(), img)
            cv.imwrite(Path(config.SN_test_path).joinpath(img_fname).as_posix(), SN_img)
            cv.imwrite(Path(config.GT_test_path).joinpath(GT_fname).as_posix(), GT)
            cv.imwrite(Path(config.marked_test_path).joinpath(mark_fname).as_posix(), mark_img)
               
    # Defining for loop to extract patches from images and GTs 
    for i in range(len(train_idx)):
        img_fname = images[train_idx[i]].parts[-1].split(".")[0].split("_")
        img = cv.imread(images[train_idx[i]].as_posix())
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        GT_fname = GT_imgs[train_idx[i]].parts[-1].split(".")[0].split("_")[0:4]
        GT = cv.imread(GT_imgs[train_idx[i]].as_posix())
        
        if (img_fname[2] == GT_fname[2] and img_fname[3] == GT_fname[3]):
            if (img.shape != (1440, 1920, 3)):
                tmp_list, GT = get_optimal_size([img], GT)
                img = tmp_list[0]
            img = get_stainNorm(config, img)
            GT[GT == 128] = 0
            x=0
            while(x<=GT.shape[0]-config.patch_size):
                y=0
                while(y<=GT.shape[1]-config.patch_size):
                    img_patch=img[x:x+config.patch_size,y:y+config.patch_size,:]
                    img_patch=cv.cvtColor(img_patch,cv.COLOR_RGB2BGR)
                    number_img=get_patchNum(config.patch_num)
                    img_patch_fname='_'
                    img_patch_fname=img_patch_fname.join(GT_fname)+"_MP_"+number_img+".png"
                    cv.imwrite(Path(patch_path).joinpath(img_patch_fname).as_posix(),img_patch)
                    
                    mask_patch=GT[x:x+config.patch_size,y:y+config.patch_size,:]
                    cv.imwrite(Path(mask_path).joinpath(img_patch_fname).as_posix(),mask_patch)
                    
                    config.patch_num += 1
                    y+=config.patch_size
                
                x+=config.patch_size
    
    return config


def process_amb_imgs(config, imgs, marked_imgs):
    for i in range(len(imgs)):
        img_fname = imgs[i].parts[-1]
        img = cv.imread(imgs[i].as_posix())
        SN_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            
        mark_fname = marked_imgs[i].parts[-1]
        mark_img = cv.imread(marked_imgs[i].as_posix())
            
        # Pre-process the ambiguous images
        if (img.shape != (1440, 1920, 3)):
            img = cv.resize(img, (1920,1440), interpolation=cv.INTER_LINEAR) 
            SN_img = cv.resize(SN_img, (1920,1440), interpolation=cv.INTER_LINEAR) 
            mark_img = cv.resize(mark_img, (1920,1440), interpolation=cv.INTER_LINEAR) 
        SN_img = get_stainNorm(config, SN_img)
        SN_img = cv.cvtColor(SN_img,cv.COLOR_RGB2BGR)
            
        # Save the pre-processed ambiguous images and their marked images to designated folders
        cv.imwrite(Path(config.amb_images).joinpath(img_fname).as_posix(), img)
        cv.imwrite(Path(config.amb_SN_images).joinpath(img_fname).as_posix(), SN_img)
        cv.imwrite(Path(config.amb_mark_images).joinpath(mark_fname).as_posix(), mark_img)