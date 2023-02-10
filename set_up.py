# %%
from pathlib import Path
import glob
import radiant_mlhub
import osgeo
from osgeo import gdal
from radiant_mlhub import Dataset
from shutil import copyfile
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os 

from utils.augmentation_functions import normalize_per_band_and_scene, normalize_per_global_band_percentiles
from utils.augmentation_functions import a_brightness, a_adjust_contrast, a_adjust_sharpness, a_resized_crop, a_hor_flip, a_ver_flip
from utils.general_utils import create_mem_ds, copy_mem_ds

import torchvision.transforms.functional as TF
import torchvision.transforms as tran
from PIL import Image

## set all seeds that you can find ;)
torch.use_deterministic_algorithms(True)
random.seed(20)
torch.manual_seed(20)
np.random.seed(20)

## globals 
dataset_id = 'nasa_rwanda_field_boundary_competition'
project_path = Path.home()  / 'projects' / 'field_delination' 

## download data?
download_data = False

datasets = Dataset.list()
#for dataset in datasets[0:5]:  # print first 5 datasets, for example
#    print(dataset)

## download data
if download_data:
    nasa_rwanda_field_boundary_competition = Dataset.fetch_by_id(dataset_id)
    nasa_rwanda_field_boundary_competition.download(output_dir=project_path / 'data')


# %% stack images
root_pth_scenes =     f'{project_path}/data/{dataset_id}/{dataset_id}_source_train'
root_pth_scenes_out = f'{project_path}/augmented_data/source_train'
root_pth_masks =      f'{project_path}/data/{dataset_id}/{dataset_id}_labels_train'
root_pth_masks_out =  f'{project_path}/augmented_data/labels_train'

folders = sorted([d for d in next(os.walk(root_pth_scenes))[1]])


## define Augmentation routines 
## for each scene id same augmentation 
aug_1 = {'rot': 1, 'brightness': 1, 'adjust_contrast': 1,'adjust_sharpness': 1, 
            'resize_crop': [10,10,236,236], 'hor_flip': True, 'ver_flip': True, 'aug_name': "_a62_"}

aug_2 = {'rot': 2, 'brightness': 1.1, 'adjust_contrast': 0.9,'adjust_sharpness': 1.05, 
            'resize_crop': [20,20,226,226], 'hor_flip': False, 'ver_flip': True, 'aug_name': "_a63_"}
    
aug_3 = {'rot': 3, 'brightness': 1.1, 'adjust_contrast': 0.85,'adjust_sharpness': 0.95, 
            'resize_crop': [5,5,220,220], 'hor_flip': True, 'ver_flip': False, 'aug_name': "_a64_"}
    
aug_4 = {'rot': 2, 'brightness': 0.95, 'adjust_contrast': 1.1,'adjust_sharpness': 1.2, 
            'resize_crop': [30,30,200,200], 'hor_flip': True, 'ver_flip': True, 'aug_name': "_a65_"}
    


# %%
# stack original images in new augmentation folder 
for folder in folders:
    folder_out = folder[-23:-8]
    scene_id   = folder_out[-10:-8]
    date_time  = folder[-7:]

    print(folder_out)
   
    full_path_scenes_in  = f'{root_pth_scenes}/{folder}'
    full_path_scenes_out = f'{root_pth_scenes_out}/{folder_out}'

    files = sorted(glob.glob(full_path_scenes_in+r'/*.tif'))
    op = osgeo.gdal.BuildVRTOptions(separate=True)

    if not os.path.isdir(full_path_scenes_out):
        os.makedirs(full_path_scenes_out)
   
    ## _a60_ --> original scene 
    full_file_path_scenes_out = full_path_scenes_out + '/'+ folder_out +'_' + date_time + '_a60_.tif'

    vrt = gdal.BuildVRT(full_path_scenes_out + '/' + folder_out +'_'+date_time+'.vrt', files, options=op)
    del vrt 
    options = { 'format': 'GTiff', 'outputType': gdal.GDT_Int16}
    tif = gdal.Translate(full_file_path_scenes_out, full_path_scenes_out + '/' + folder_out +'_'+date_time+'.vrt', **options  )
    del tif 

     ## copy mask 
    mask_folder_in  = folder.replace("source", "labels")
    mask_folder_in  = mask_folder_in[:-8]
    mask_folder_out = mask_folder_in[-15:]
    
    full_path_mask_in  = f'{root_pth_masks}/{mask_folder_in}'
    full_path_mask_out = f'{root_pth_masks_out}/{mask_folder_out}'

    if not os.path.isdir(full_path_mask_out):
        os.makedirs(full_path_mask_out)

    mask_file_path_out = f"{full_path_mask_out}/{mask_folder_out}_{date_time}_a60_.tif"
    copyfile(f"{full_path_mask_in}/raster_labels.tif", mask_file_path_out )

    

# %%
for folder in folders:

    print(folder)
    folder_out = folder[-23:-8]
    scene_id   = folder[-10:-8]

    print(scene_id)
 
    date_time  = folder[-7:]
    print(folder)
    print(folder_out)
    print(date_time)

    mask_folder_in  = folder.replace("source", "labels")
    mask_folder_in  = mask_folder_in[:-8]
    mask_folder_out = mask_folder_in[-15:]
    full_path_mask_augmentation = f'{root_pth_masks_out}/{mask_folder_out}'
    full_path_scenes_augmentaion = f'{root_pth_scenes_out}/{folder_out}/{folder_out}_{date_time}_a60_.tif'
    mask_file_path_augmentation = f"{ full_path_mask_augmentation}/{mask_folder_out}_{date_time}_a60_.tif"
    ## construct augmentations
    #original_scene = torch.from_numpy(rio.open(full_path_scenes_augmentaion).read())
    #original_mask  = torch.from_numpy(rio.open(mask_file_path_augmentation).read())

    original_scene = rio.open(full_path_scenes_augmentaion).read()
    original_mask  = rio.open(mask_file_path_augmentation).read()

    #original_scene = (original_scene / 10000) * 32767
    #print(original_scene[0,1:8,1:8])

    ## normalize per band 
    #original_scene = normalize_per_band_and_scene(original_scene) 
    #print(original_scene.shape)
    original_scene = normalize_per_global_band_percentiles(original_scene, 4)

    ## write normalized scene
    mem_img_aug = create_mem_ds(full_path_scenes_augmentaion, 4)

    # write bands
    for b in range(original_scene.shape[0]):
       mem_img_aug.GetRasterBand(b+1).WriteArray(original_scene[b,:,:]) 
    
    full_path_scenes_augmentaion_nor = f'{root_pth_scenes_out}/{folder_out}/{folder_out}_{date_time}_a61_.tif'
    copy_mem_ds(full_path_scenes_augmentaion_nor, mem_img_aug)

    # write mask
    mem_img_msk = create_mem_ds(mask_file_path_augmentation, 1)
    mem_img_msk.GetRasterBand(1).WriteArray(original_mask[0,:,:]) 
        
    full_path_scenes_augmentaion_mask = f'{ full_path_mask_augmentation}/{mask_folder_out}_{date_time}_a61_.tif'
    copy_mem_ds(full_path_scenes_augmentaion_mask, mem_img_msk)

    original_scene = torch.from_numpy(original_scene)
    original_mask  = torch.from_numpy(original_mask)

    Augs = [aug_1, aug_2, aug_3, aug_4]
    for u, aug in enumerate(Augs):
        print("start augmentation with:", aug['aug_name'])

        scene = original_scene
        mask = original_mask
        scene = torch.rot90(original_scene, aug['rot'], (1,2))
        mask  = torch.rot90(original_mask, aug['rot'], (1,2))

        

        ## brightness
        scene = a_brightness(scene, aug['brightness'], 4) 
        ## sharpness
        scene = a_adjust_sharpness(scene, aug['adjust_sharpness'], 4)
        ## contrast
        scene = a_adjust_contrast(scene, aug['adjust_contrast'], 4)
        ## resize
        scene = a_resized_crop(scene, 4, aug['resize_crop'][0],aug['resize_crop'][1],
            aug['resize_crop'][2],aug['resize_crop'][3], 256, tran.InterpolationMode.NEAREST)
        mask = a_resized_crop(mask, 1, aug['resize_crop'][0],aug['resize_crop'][1],
            aug['resize_crop'][2],aug['resize_crop'][3], 256, tran.InterpolationMode.NEAREST)
        ## flips 
        if(aug['hor_flip']==True):
            scene = a_hor_flip(scene, 4)
            mask = a_hor_flip(mask, 1)
        if(aug['ver_flip']==True):
            scene = a_ver_flip(scene, 4)
            mask = a_ver_flip(mask, 1)

        ## write 
        print(full_path_scenes_augmentaion)
        print(mask_file_path_augmentation)
        mem_img_aug = create_mem_ds(full_path_scenes_augmentaion, 4)
        mem_img_msk = create_mem_ds(mask_file_path_augmentation, 1)

        # write bands
        for b in range(original_scene.shape[0]):
            mem_img_aug.GetRasterBand(b+1).WriteArray(scene[b,:,:].numpy()) 
        mem_img_msk.GetRasterBand(1).WriteArray(mask[0,:,:].numpy()) 


        scene_id_folder_name = f"source_train_{scene_id}"
        nn = aug['aug_name']
        full_path_scenes_augmentaion_sceme = f'{root_pth_scenes_out}/{scene_id_folder_name}/{folder_out}_{date_time}{nn}.tif'
        print(full_path_scenes_augmentaion_sceme)
        copy_mem_ds(full_path_scenes_augmentaion_sceme, mem_img_aug)
        full_path_scenes_augmentaion_mask = f'{ full_path_mask_augmentation}/{mask_folder_out}_{date_time}{nn}.tif'
        copy_mem_ds(full_path_scenes_augmentaion_mask, mem_img_msk)

        """
        ## show original scene and augmented scene
        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(121)  # left side
        ax2 = fig.add_subplot(122)  # right side
        #ax1.imshow(field.T.astype('int8'), vmin=0, vmax=255)  # rgb band
        ax1.imshow(original_scene[0:3,:,:].permute(1, 2, 0))  # rgb band 
        ax2.imshow(scene[0:3,:,:].permute(1, 2, 0))
        plt.tight_layout()
        plt.show()

         ## show original scene and augmented scene
        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(121)  # left side
        ax2 = fig.add_subplot(122)  # right side
        #ax1.imshow(field.T.astype('int8'), vmin=0, vmax=255)  # rgb band
        ax1.imshow(scene[0:3,:,:].permute(1, 2, 0))  # rgb band 
        ax2.imshow(mask.permute(1, 2, 0))
        plt.tight_layout()
        plt.show()

        
        print(scene[0,1:8,1:8])
        print(original_scene[1,1:8,1:8])
        print(original_scene[2,1:8,1:8])
        print(original_scene[3,1:8,1:8])
        
        print(scene[0,1:8,1:8])
        print(scene.shape)
        
        ## histogram
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        fig.suptitle('band-hist of normalized scene')
        ax1.hist(scene[0,:,:])
        ax1.set_title('Blue')
        ax2.hist(scene[1,:,:])
        ax2.set_title('Green')
        ax3.hist(scene[2,:,:])
        ax3.set_title('Red')
        ax4.hist(scene[3,:,:])
        ax4.set_title('NIr')

        plt.show()
        print(folder)
        """
        print("finished... ")




# %% old
# scale to 255
scale_uint8 = False
if scale_uint8 == True:
    dd = rio.open(full_file_path_scenes_out).read()
    dd = dd / 10000 * 256
    with rio.open(full_file_path_scenes_out, mode="w",
        driver="GTiff",
        height=256,
        width=256,
        count = 4,
        dtype=rio.uint8) as dst:
        dst.write(dd.astype(rio.uint8))
    
    dd = rio.open(full_file_path_scenes_out).read()