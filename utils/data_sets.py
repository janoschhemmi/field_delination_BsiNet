# %%
import torch
import numpy as np
import cv2
from PIL import Image, ImageFile
# %%

import skimage
from skimage import io
import imageio
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
import os
from osgeo import gdal
import rasterio as rio

# %%

## read tif 
def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + " no data ")
    #  
    width = dataset.RasterXSize
    #  
    height = dataset.RasterYSize
    #  n bands
    bands = dataset.RasterCount
    #  if not set, copy from input data 
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    # 
    geotrans = dataset.GetGeoTransform()
    #  
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj



#write gdal
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
          im_bands, (im_height, im_width) = 1, im_data.shape
    # create
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # set geotrans
        dataset.SetProjection(im_proj)  # set proj
    if im_bands == 1:
      dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
           dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


## dataset tripple 
class DatasetImageMaskContourDist(Dataset):

    def __init__(self, dir, file_names, distance_type):

        self.dir = dir
        self.file_names = file_names
        self.distance_type = distance_type

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        image =   load_image(os.path.join(self.dir,img_file_name+'.tif'))
        mask =    load_mask(os.path.join(self.dir,img_file_name+'.tif'))
        contour = load_contour(os.path.join(self.dir,img_file_name+'.tif'))
        dist =    load_distance(os.path.join(self.dir,img_file_name+'.tif'), self.distance_type)

        #print(image[0,1:10,1:10])
        #print(mask[0,1:10,1:10])
        #print(contour[0,1:10,1:10])
        #print(dist)

        return img_file_name, image, mask, contour, dist


class DatasetImageMaskContourDist_test(Dataset):

    def __init__(self, dir, file_names, distance_type):

        self.dir = dir
        self.file_names = file_names
        self.distance_type = distance_type

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        image =    load_image(os.path.join(self.dir,img_file_name))
        #mask =    load_mask(os.path.join(self.dir,img_file_name+'.tif'))
        #contour = load_contour(os.path.join(self.dir,img_file_name+'.tif'))
        #dist =    load_distance(os.path.join(self.dir,img_file_name+'.tif'), self.distance_type)

        return img_file_name, image


def load_image(path):

    img = rio.open(path).read()


    """data_transforms = transforms.Compose(
        [
           # transforms.Resize(256),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ]
    )"""
    #print("loading_image")
    #print(np.min(img))
    #print(np.max(img))
    img = torch.tensor(img, dtype=torch.float)
    # img = torch.tensor(img)


    return img


def load_mask(path):
    mask = rio.open(path.replace("image", "mask")).read()
   # im_width, im_height, im_bands, mask, im_geotrans, im_proj = readTif(path.replace("image", "mask").replace("tif", "tif"))
    ###mask = mask/225.

    mask[mask == 255] = 1
    mask[mask == 0] = 0

    return torch.from_numpy(np.expand_dims(mask, 0)).long()


def load_contour(path):

    contour = rio.open(path.replace("image", "cont")).read()
    ###contour = contour/255.
    contour[contour ==255] = 1
    contour[contour == 0] = 0

    """print("__________________________________________________")
    print("__________________________________________________")
    print("cont check ")

    print(np.min(contour))
    print(np.max(contour))
    print(contour[0,100:120,110:130])"""

    return torch.from_numpy(np.expand_dims(contour, 0)).long()


def load_distance(path, distance_type):

    if distance_type == "dist_mask":
        path = path.replace("image", "dist")
        dist = cv2.imread(path)

    if distance_type == "dist_contour":

        #print("______________________________________")
        #print("reading distance")
        path = path.replace("image", "dist")
        dist = rio.open(path).read()
        

    if distance_type == "dist_contour_tif":
        dist = cv2.imread(path.replace("image", "dist_contour_tif"), 0)
        # dist = dist/255.

    return torch.from_numpy(dist).float()