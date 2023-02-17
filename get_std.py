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
import glob
import rasterio as rio 


file_list = glob.glob("/home/hemmerling/projects/field_delination_BsiNet/data/nasa_rwanda_field_boundary_competition/nasa_rwanda_field_boundary_competition_source_train/**/*.tif", recursive=True)
print(file_list[1:10])
print(len(file_list))
file_list = list(set(file_list))
print(len(file_list))


def strFilter(string, substr):
    return [str for str in string if
             any(sub in str for sub in substr)]


B01 = strFilter(file_list, "B01.tif")

B01 = [i for i in file_list if any(i for j in ["B01.tif", "B01.tif"] if str(j) in i)]
print(B01)
# %%
B01 = [i for i in file_list if any(i for j in ["B01.tif", "B01.tif"] if str(j) in i)]
array_list = []
for i, band in enumerate(B01):

    print(i)
    layer = rio.open(band).read()
    print(layer)
    print(layer.dtype)
    array_list.append(layer)
  
result_arr = np.vstack(array_list)

flat = result_arr.flatten()
minn_01 = np.min(flat)
maxx_01= np.max(flat)
std_01  = np.std(flat)
mean_01 = np.mean(flat)
percent_1_01 = np.percentile(flat, 0.5)
percent_99_01 = np.percentile(flat, 99.5)

print(std_01)

# %%
B02 = [i for i in file_list if any(i for j in ["B02.tif", "B02.tif"] if str(j) in i)]
array_list = []
for i, band in enumerate(B02):

    print(i)
    layer = rio.open(band).read()
    array_list.append(layer)
  
result_arr = np.vstack(array_list)

flat = result_arr.flatten()
minn_02 = np.min(flat)
maxx_02= np.max(flat)
std_02  = np.std(flat)
mean_02 = np.mean(flat)
percent_1_02 = np.percentile(flat, 0.5)
percent_99_02 = np.percentile(flat, 99.5)

# %%
B03 = [i for i in file_list if any(i for j in ["B03.tif", "B03.tif"] if str(j) in i)]
array_list = []
for i, band in enumerate(B03):

    print(i)
    layer = rio.open(band).read()
    array_list.append(layer)
  
result_arr = np.vstack(array_list)

flat = result_arr.flatten()
minn_03 = np.min(flat)
maxx_03= np.max(flat)
std_03  = np.std(flat)
mean_03 = np.mean(flat)
percent_1_03 = np.percentile(flat, 0.5)
percent_99_03 = np.percentile(flat, 99.5)

# %%
B04 = [i for i in file_list if any(i for j in ["B04.tif", "B04.tif"] if str(j) in i)]
array_list = []
for i, band in enumerate(B04):

    print(i)
    layer = rio.open(band).read()
    array_list.append(layer)
  
result_arr = np.vstack(array_list)

flat = result_arr.flatten()
minn_04 = np.min(flat)
maxx_04= np.max(flat)
std_04  = np.std(flat)
mean_04 = np.mean(flat)
percent_1_04 = np.percentile(flat, 0.5)
percent_99_04 = np.percentile(flat, 99.5)

# %%

print(std_01)
print(mean_01)

print(std_02)
print(mean_02)

print(std_03)
print(mean_03)

print(std_04)
print(mean_04)

print(percent_1_01,percent_1_02,percent_1_03,percent_1_04)
print(percent_99_01,percent_99_02,percent_99_03,percent_99_04)

#  write into csv 
# %%
import csv
with open('/home/hemmerling/projects/field_delination_BsiNet/stats_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["band", "mean", "std"])
    writer.writerow([1, mean_01, std_01])
    writer.writerow([2, mean_02, std_02])
    writer.writerow([3, mean_03, std_03])
    writer.writerow([4, mean_04, std_04])
# %%
check = True
if check == True:

    import torch
    import numpy as np
    import cv2
    from PIL import Image, ImageFile

    import skimage
    from skimage import io
    import imageio
    from torch.utils.data import Dataset
    from torchvision import transforms
    from scipy import io
    import os
    from osgeo import gdal
    import glob
    import rasterio as rio 

    file_list = glob.glob("/home/hemmerling/projects/field_delination_BsiNet/data_preprocessed/train/image/*.tif", recursive=True)
    print(file_list[1:10])
    #print(len(file_list))
    file_list = list(set(file_list))
    print(len(file_list))

   
    test = [i for i in file_list if any(i for j in ["a60_.tif", "a60_.tif"] if str(j) in i)]


    array_list = []
    for i, band in enumerate(test):

        print(i)
        layer = rio.open(band).read()
        print(layer)
        print(layer.dtype)
        array_list.append(layer)
  
    result_arr = np.vstack(array_list)

    flat = result_arr.flatten()
    minn_test = np.min(flat)
    maxx_test= np.max(flat)
    std_test  = np.std(flat)
    mean_test = np.mean(flat)


print(minn_test)
print(maxx_test)
print(mean_test)
print(std_test)


# %%
