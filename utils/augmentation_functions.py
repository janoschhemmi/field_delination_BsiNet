# %%
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path
import rasterio as rio
import torch
import matplotlib as plt
import torchvision
import random
import torch 
import torchvision.transforms.functional as TF
import torchvision.transforms as tran
import cv2

torch.use_deterministic_algorithms(True)
random.seed(20)
torch.manual_seed(20)

# %%

def a_brightness( 
    tensor,
    brightness : 0.5,
    n_bands: 4
):  
    """ 0 --> black image
        1 --> original image 
        2 --> factor 2  """

    for i in range(n_bands):

        layer = tensor[:,:,i]
        layer = TF.adjust_brightness(layer, brightness_factor = brightness)

        if i == 0:
            out = layer 
        if i == 1:
            out_1 = torch.stack((out, layer))
    
        if i == 2:
            out_2 = layer 
        if i == 3:
            out_3 = torch.stack((out_2, layer))
            out = torch.cat((out_1, out_3), dim = 0)
    return(out)


def a_adjust_contrast(
    tensor,
    factor : 1,
    n_bands: 4
    ):

    """ 0 --> black image
        1 --> original image 
        2 --> factor 2  """

    for i in range(n_bands):
        
   
        layer = tensor[i,:,:]
        layer = layer[None, :, :]
        layer = TF.adjust_contrast(layer, contrast_factor = factor)

        if i == 0:
            out = layer 
        if i == 1:
            out_1 = torch.stack((out, layer))
       
        if i == 2:
            out_2 = layer 
        if i == 3:
            out_3 = torch.stack((out_2, layer))
            out = torch.cat((out_1, out_3), dim = 0)
            out = out[:, -1, :,:]
    return(out)


def a_adjust_sharpness(
    tensor,
    factor : 1,
    n_bands: 4
    ):

    """ 0 --> black image
        1 --> original image 
        2 --> factor 2  """

    for i in range(n_bands):

        layer = tensor[i,:,:]
        layer = layer[None, :, :]

        #layer = layer[None, :, :]
        layer = TF.adjust_sharpness(layer, sharpness_factor=factor)

        if i == 0:
            out = layer 
        if i == 1:
            out_1 = torch.stack((out, layer))
       
        if i == 2:
            out_2 = layer 
        if i == 3:
            out_3 = torch.stack((out_2, layer))
            out = torch.cat((out_1, out_3), dim = 0)
            out = out[:, -1, :,:]
    return(out)



def a_resized_crop(tensor, n_bands, i, j, h, w, size, interpolation='BILINEAR'):
    """Crop the given CV Image and resize it to desired size. Notably used in RandomResizedCrop.

    Args:
        img (np.ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (str, optional): Desired interpolation. Default is
            ``BILINEAR``.
    Returns:
        np.ndarray: Cropped image.
    """
    for i in range(n_bands):

   
        layer = tensor[i,:,:]
        layer = layer[None, :, :]

        #layer = layer[None, :, :]

        layer = TF.resized_crop(layer, i,j,h,w, size, interpolation=interpolation )
 
        if i == 0:
            out = layer 
        if i == 1:
            out_1 = torch.stack((out, layer))
       
        if i == 2:
            out_2 = layer 
        if i == 3:
            out_3 = torch.stack((out_2, layer))
            out = torch.cat((out_1, out_3), dim = 0)
            out = out[:, -1, :,:]
            print(out.shape)
    print("out")
    print(out.shape)
    return out         


def a_hor_flip(
    tensor,
    n_bands
):
    for i in range(n_bands):
   
        layer = tensor[i,:,:]
        layer = layer[None, :, :]

        #layer = layer[None, :, :]
        layer = TF.hflip(layer)

        if i == 0:
            out = layer 
        if i == 1:
            out_1 = torch.stack((out, layer))
       
        if i == 2:
            out_2 = layer 
        if i == 3:
            out_3 = torch.stack((out_2, layer))
            out = torch.cat((out_1, out_3), dim = 0)
            out = out[:, -1, :,:]

    return out         



def a_ver_flip(
    tensor,
    n_bands
):
    for i in range(n_bands):
   
        layer = tensor[i,:,:]
        layer = layer[None, :, :]

        #layer = layer[None, :, :]
        layer = TF.vflip(layer)

        if i == 0:
            out = layer 
        if i == 1:
            out_1 = torch.stack((out, layer))
       
        if i == 2:
            out_2 = layer 
        if i == 3:
            out_3 = torch.stack((out_2, layer))
            out = torch.cat((out_1, out_3), dim = 0)
            out = out[:, -1, :,:]

    return out

    # %%    


def normalize_per_band_and_scene(
    array: np.ndarray
):
    """ normalise image to give a meaningful output """
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def normalize_per_global_band_percentiles(
    array: np.ndarray,
    n_bands
):
    for i in range(n_bands):
   
        layer = array[i,:,:]
        print(layer)
   

        if i == 0:
            
            array_min, array_max = 324.0, 656.0
            layer = (layer - array_min) / (array_max - array_min)
            out = layer

        if i == 1:
            array_min, array_max = 610.0, 1062.0
            layer = (layer - array_min) / (array_max - array_min)
            out = np.stack((out, layer))
            
        if i == 2:
            array_min, array_max = 529.0, 1458.0
            layer = (layer - array_min) / (array_max - array_min)
            layer_2 = layer[None, :, :]

        if i == 3:
            array_min, array_max = 1745.0, 4319.0
            layer = (layer - array_min) / (array_max - array_min)
            layer_3 = layer[None, :, :]
            out_2 = np.stack((layer_2, layer_3))
            out_2 = out_2[:,-1,:,:]
            out = np.concatenate((out, out_2),axis = 0 )

    return out
  
# %%
def apply_augmentation_on_tiles(
    tile: str = "",
    train_source_items:  str = "",
    dataset_id: str = "",

): 

    bd1 = rio.open(f"{train_source_items}/{dataset_id}_source_train_{tile}/B01.tif")
    bd1_array = bd1.read(1)
    bd2 = rio.open(f"{train_source_items}/{dataset_id}_source_train_{tile}/B02.tif")
    bd2_array = bd2.read(1)
    bd3 = rio.open(f"{train_source_items}/{dataset_id}_source_train_{tile}/B03.tif")
    bd3_array = bd3.read(1)
    bd4 = rio.open(f"{train_source_items}/{dataset_id}_source_train_{tile}/B04.tif")
    bd4_array = bd4.read(1)
    b01_norm = normalize_per_band_and_scene(bd1_array)
    b02_norm = normalize_per_band_and_scene(bd2_array)
    b03_norm = normalize_per_band_and_scene(bd3_array)
    b04_norm = normalize_per_band_and_scene(bd4_array)

    ids_list  = tile.split('_') # XX_YYYY_MM where XX is the training file id and YYYY_MM is the timestamp
    tile_id   = ids_list[0]
    timestamp = f"{ids_list[1]}_{ids_list[2]}"

    field = np.dstack((b04_norm, b03_norm, b02_norm, b01_norm))
    mask  = rio.open(Path.cwd() / f"{train_label_items}/{dataset_id}_labels_train_{tile_id}/raster_labels.tif").read(1) 

    #create a folder for the augmented images
    if not os.path.isdir(f"./augmented_data/{timestamp}"):
        os.makedirs(f"./augmented_data/{timestamp}")
    if not os.path.isdir(f"./augmented_data/{timestamp}/fields"):
        os.makedirs(f"./augmented_data/{timestamp}/fields")
    if not os.path.isdir(f"./augmented_data/{timestamp}/masks"):
        os.makedirs(f"./augmented_data/{timestamp}/masks")

    main( #applying augmentation effects
        field  = field,
        mask   = mask,
        prefix = tile_id,
        write_folder = f"./augmented_data/{timestamp}"
    ) #approximately 30 seconds
# %%
## MASK

def get_boundary(label, kernel_size = (1,1)):
    tlabel = label.astype(np.uint8)
    temp = cv2.Canny(tlabel,0,1)
    tlabel = cv2.dilate(
        temp,
        cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            kernel_size),
        iterations = 1)
    tlabel = tlabel.astype(np.float32)
    tlabel /= 255.
    return tlabel

def get_crop(image, kernel_size = (3,3)):

    im_floodfill = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # floodfill
    cv2.floodFill(im_floodfill, mask, (0,0), 1)

    # invert
    im_floodfill = cv2.bitwise_not(im_floodfill)

    # kernel size
    kernel = np.ones(kernel_size, np.uint8)

    # erode & dilate
    img_erosion = cv2.erode(im_floodfill, kernel, iterations=1)
    cont =  cv2.dilate(img_erosion, kernel, iterations=1) - 254
    return cont[np.newaxis,:,:]

def get_distance(label):

    tlabel = label.transpose(1,2,0).astype(np.uint8)

 
    dist = cv2.distanceTransform(tlabel,
                                 cv2.DIST_L1,
                                 0)

    dist = dist[:,:,np.newaxis]

    # get unique objects
    output = cv2.connectedComponentsWithStats(label.transpose(1,2,0), 4, cv2.CV_32S)


    num_objects = output[0]
    labels = output[1]
    
    # min/max normalize dist for each object
    for l in range(num_objects):
      dist[labels==l] = (dist[labels==l]) / (dist[labels==l].max())

    return dist[np.newaxis,:,:,-1]

# %%