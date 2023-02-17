# %%
import torch
from tqdm import tqdm
import numpy as np
import torchvision
from torch.nn import functional as F
import time
import argparse

# %%
def evaluate(device, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):

            _, inputs, targets, _,_ = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            #print("ee")
            #print(outputs[0].shape)
            #print(targets.squeeze(1).shape)
            #print(targets[:,-1,:,:].squeeze(1).shape)
            loss = F.nll_loss(outputs[0], targets[:,-1,:,:].squeeze(1))
            losses.append(loss.item())

        writer.add_scalar("Dev_Loss", np.mean(losses), epoch)

    return np.mean(losses), time.perf_counter() - start

# %%
def visualize(device, epoch, model, data_loader, writer, val_batch_size, train=True):
    def save_image(image, tag, val_batch_size):
        from PIL import Image
     

        
        #print(image.shape)
        image = image[-1,-1,:,: ]#.permute(1,2,0)
        #print(image.shape)
        #print(image.min())
        #print(image.max())
        image -= image.min()
        image /= image.max()
        image  = image * 255 ## evtl nonsense
        #print(image.min())
        #print(image.max())
        grid = torchvision.utils.make_grid(
            image * 255, nrow=int(np.sqrt(val_batch_size)), pad_value=0, padding=25
        )
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _, inputs, targets, _,_ = data

            inputs = inputs.to(device)

            targets = targets.to(device)
            outputs = model(inputs)

            output_mask = outputs[0].detach().cpu().numpy()
            output_final = np.argmax(output_mask, axis=1).astype(float)
            output_final = torch.from_numpy(output_final).unsqueeze(1)

            if train == "True":
                save_image(targets.float(), "Target_train",val_batch_size)
                save_image(output_final, "Prediction_train",val_batch_size)
            else:
                save_image(targets.float(), "Target", val_batch_size)
                save_image(output_final, "Prediction", val_batch_size)

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
  copy_ds = gdal.GetDriverByName("GTiff").CreateCopy(pth, mem_ds, 0, options=['COMPRESS=LZW'])
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
