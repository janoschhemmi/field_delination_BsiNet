# %%
import torch
from tqdm import tqdm
import numpy as np
import torchvision
from torch.nn import functional as F
import time
import argparse
import torch.nn as nn
from utils.losses import dice_loss
from utils.losses import LossBsiNet


def strFilter(string, substr):
    return [str for str in string if
             any(sub in str for sub in substr)]


# %%
def evaluate(device, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):

            _, inputs, targets_1, targets_2,targets_3 = data
            inputs  = inputs.to(device)
            targets_1 = targets_1.to(device)
            targets_2 = targets_2.to(device)
            targets_3 = targets_3.to(device)
            outputs = model(inputs)
            #print("_________________________-")
            #print("inside val loss ")
            #print(len(outputs))
            #print(len(targets_1))

            criterion = LossBsiNet([1,1,1], device=device)
            loss = criterion(
                outputs[0], outputs[1], outputs[2], targets_1, targets_2, targets_3, epoch, writer
            )

            #print(outputs[0].size())
            #print(targets[0].size())
        
            #print(targets.squeeze(1).shape)
            #print(targets[:,-1,:,:].squeeze(1).shape)
            #loss = dice_loss(outputs[0], targets[0]) #.squeeze(1))
            #loss_fun = nn.BCEWithLogitsLoss()
            #loss = loss_fun(outputs[0].flatten(), targets[0].flatten().float())
            print("val loss: ", loss)
            losses.append(loss.item())

        writer.add_scalar("Val_Loss", np.mean(losses), epoch)

    return np.mean(losses), time.perf_counter() - start

# %%
def visualize(device, epoch, model, data_loader, writer, val_batch_size, train=True):
    def save_image(image, tag, val_batch_size):
        from PIL import Image
     
        #print("______________________________")
        #print("inside visualize", tag)
        #print(image.shape)
        #image = image[-1,-1,:,: ]#.permute(1,2,0)
        #print("shape: ",image.shape)
        #print(image.min())
        #print(image.max())
        #print(type(image))
        #print(image.dtype())

        #print(image[1,0,0,1:10,1:10])
        
        
        
        #image  = image * 255 ## evtl nonsense

        #print("after norm")
        #print(image.min())
        #print(image.max()) # 4D mini-batch Tensor of shape (B x C x H x W)
        
        #grid = torchvision.utils.make_grid(
        #    image * 255, nrow=int(np.sqrt(val_batch_size)), pad_value=0, padding=25
        #)
        #print(grid)
        if tag == "Target":
            
            #print("###########################")
            #print("inside target:")
            #print(image.shape)

        
          
            image_s = image[0]
            #image -= image.min()
            #image /= image.max()
           
            image_s = image_s.astype(float)
            cont = image[1]
            dist = image[2]
            #print(image_s.shape)
            #print(cont.shape)
            #print(dist.shape)
            #print(len(image))
            #print("shape of target:", dist.shape)
            #print("min of target:", dist.min())
            #print("max of target:", dist.max())
            #print(image[100:110,100:110])
            writer.add_image(tag, image_s[0,0,:,:], epoch, dataformats='CHW')
            writer.add_image("Ref_contour", cont[0,0,:,:] , epoch, dataformats='CHW')
            writer.add_image("REf_dist", dist[0,0,:,:], epoch, dataformats='HW')

       
        if tag == "Input":
            image = image.astype(float)
            writer.add_image(tag, image[0,:,:,:], epoch, dataformats='CHW')
        if tag == "Prediction_dist":
            image = image.astype(float)
            image -= image.min()
            image /= image.max()
            writer.add_image(tag, image[0,-1,:,:], epoch, dataformats='HW')
        
        if tag == "Prediction_cont":
            image = image.astype(float)
            image -= image.min()
            image /= image.max()
            
            writer.add_image(tag, image[0,-1,:,:], epoch, dataformats='HW')
    
        if tag == "Prediction_mask":
            image = image.astype(float)
            image -= image.min()
            image /= image.max()
            writer.add_image("prediction mask binary", np.where(image[0,-1,:,:]> 0.5,1,0), epoch, dataformats='HW')
            writer.add_image(tag, image[0,-1,:,:], epoch, dataformats='HW')
       
    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _, inputs, targets_1, targets_2, targets_3 = data

            inputs = inputs.to(device)
            targets_1 = targets_1.detach().cpu().numpy()
            targets_2 = targets_2.detach().cpu().numpy()
            targets_3 = targets_3.detach().cpu().numpy()
            
            outputs = model(inputs)
            #print("!! went through model")
            #print(len(outputs))
            #print(outputs[0].shape)
            #print(outputs[1].shape)
            #print(outputs[2].shape)


            output_mask = outputs[0].detach().cpu().numpy()
            output_cont = outputs[1].detach().cpu().numpy()
            output_dist = outputs[2].detach().cpu().numpy()


            #output_final = np.argmax(output_mask, axis=1).astype(float)
            #output_final = torch.from_numpy(output_final).unsqueeze(1)

            #if train == "True":
                #save_image(targets, "Target_train", val_batch_size)
                #save_image(output_final, "Prediction_train",val_batch_size)
            
            save_image(inputs.detach().cpu().numpy(), "Input", val_batch_size)
            save_image([targets_1, targets_2, targets_3], "Target", val_batch_size)
            save_image(output_mask, "Prediction_mask", val_batch_size)
            save_image(output_cont, "Prediction_cont", val_batch_size)
            save_image(output_dist, "Prediction_dist", val_batch_size)


            break

# %% 
def create_train_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--train_path", type=str, help="path to img tif files")
    parser.add_argument("--val_path", type=str, help="path to img tif files")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: bsinet",
    )
    parser.add_argument("--object_type", type=str, help="Dataset.")
    parser.add_argument(
        "--distance_type",
        type=str,
        default="dist_contour",
        help="select distance transform type - dist_mask,dist_contour,dist_contour_tif",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="train batch size")
    parser.add_argument(
        "--val_batch_size", type=int, default=4, help="validation batch size"
    )
    parser.add_argument("--num_epochs", type=int, default=150, help="number of epochs")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument(
        "--use_pretrained", type=bool, default=False, help="Load pretrained checkpoint."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="If use_pretrained is true, provide checkpoint.",
    )
    parser.add_argument("--save_path", type=str, help="Model save path.")

    return parser


def create_validation_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: bsinet",
    )
    parser.add_argument("--test_path", type=str, help="path to img tif files")
    parser.add_argument("--model_file", type=str, help="model_file")
    parser.add_argument("--save_path", type=str, help="results save path.")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")

    return 
# %%


import os
import rasterio as rio 
import numpy as np 
from pathlib import Path
import matplotlib as plt 
import gdal


def clean_string(s: str, dataset_id: str) -> str:
    """
    extract the tile id and timestamp from a source image folder
    e.g extract 'ID_YYYY_MM' from 'nasa_rwanda_field_boundary_competition_source_train_ID_YYYY_MM'
    """
    s = s.replace(f"{dataset_id}_source_", '').split('_')[1:]
    return '_'.join(s)

# create dataset in memory using geotransform specified in ref_pth
def create_mem_ds(ref_pth, n_bands, dtype=gdal.GDT_Float32):
  print('creating empty raster \n copying geotransform of' + ref_pth)
  drvMemR = gdal.GetDriverByName('MEM')
  ds = gdal.Open(ref_pth)
  mem_ds = drvMemR.Create('', ds.RasterXSize, ds.RasterYSize, n_bands, dtype)
  mem_ds.SetGeoTransform(ds.GetGeoTransform())
  mem_ds.SetProjection(ds.GetProjection())
  return mem_ds

# create copy
def copy_mem_ds(pth, mem_ds):
  copy_ds = gdal.GetDriverByName("GTiff").CreateCopy(pth, mem_ds, 0) #, options=['COMPRESS=LZW'])
  copy_ds = None

  
def read_bands_and_mask_of_scene(
    train_source_scenes_path = str,
    dataset_id = str,
    scene_id_tile_and_time = str,
    train_label_path = str
):
    """
    returns stacked scene and ref mask 
    """
    from utils.augmentation_functions import normalize_per_band_and_scene

    path_to_scene = f"{train_source_scenes_path}/{dataset_id}_source_train_{scene_id_tile_and_time}"
    bd1 = rio.open(f"{path_to_scene}/B01.tif")
    bd1_array = bd1.read(1)
    bd2 = rio.open(f"{path_to_scene}/B02.tif")
    bd2_array = bd2.read(1)
    bd3 = rio.open(f"{path_to_scene}/B03.tif")
    bd3_array = bd3.read(1)
    bd4 = rio.open(f"{path_to_scene}/B04.tif")
    bd4_array = bd4.read(1)
    b01_norm = normalize_per_band_and_scene(bd1_array)
    b02_norm = normalize_per_band_and_scene(bd2_array)
    b03_norm = normalize_per_band_and_scene(bd3_array)
    b04_norm = normalize_per_band_and_scene(bd4_array)

    field = np.dstack((b04_norm, b03_norm, b02_norm, b01_norm))
    mask  = rio.open(Path.cwd() / f"{train_label_path}/{dataset_id}_labels_train_{scene_id_tile_and_time.split('_')[0]}/raster_labels.tif").read(1)
    return field, mask 


def name_to_id_and_time_stamp(
    tile: str = "",
    ):
    ids_list  = tile.split('_') # XX_YYYY_MM where XX is the training file id and YYYY_MM is the timestamp
    tile_id   = ids_list[0]
    timestamp = f"{ids_list[1]}_{ids_list[2]}"
    return tile_id, timestamp

def stack_bands(file_list):
    with rio.open(file_list[0]) as src:
        stacked_image = src.read(1)
    for file in file_list[1:]:
        with rio.open(file) as src:
            stacked_image = np.dstack((stacked_image, src.read(1)))
    return stacked_image
    original = None
# 

def show_image_scene_and_mask(field:np.ndarray, mask:np.ndarray ): 
    """Show the field and corresponding mask."""
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    #ax1.imshow(field.T.astype('int8'), vmin=0, vmax=255)  # rgb band
    ax1.imshow((field.T / 5000 * 255).astype('uint8'))  # rgb band
    
    plt.gray()
   
    ax2.imshow(mask.T)
    plt.tight_layout()
    plt.show()
