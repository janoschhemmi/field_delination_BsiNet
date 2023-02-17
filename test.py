# %%
import torch
import os
from torch.utils.data import DataLoader
from utils.data_sets import DatasetImageMaskContourDist_test
import glob
from models.BsiNet import BsiNet_2, BsiNet
from tqdm import tqdm
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import rasterio as rio



str_to_test_on = "2021_08"

def build_model(model_type):

    if model_type == "bsinet_2":
        model = BsiNet_2(input_channels=4 ,num_classes=2)

    return model

def strFilter(string, substr):
    return [str for str in string if
             any(sub in str for sub in substr)]



if __name__ == "__main__":

    args = argparse.ArgumentParser(description="test setup for segmentation test")
    args.save_path_predictions = '/home/hemmerling/projects/field_delination_BsiNet/models_safe/predictions/'
    args.save_path_model = '/home/hemmerling/projects/field_delination_BsiNet/models_safe/'
    
    args.model_file = os.path.join(args.save_path_model, "30" + ".pt")
 
    args.model_type = 'bsinet_2'
    args.distance_type = 'dist_contour'
    args.test_path = '/home/hemmerling/projects/field_delination_BsiNet/data_preprocessed/test/image/'


    test_path = os.path.join(args.test_path)
    model_file = args.model_file
    save_path_model  = args.save_path_model
    save_path_predictions  = args.save_path_predictions

    model_type = args.model_type
    print(model_file)

    device = torch.device("cuda")

    test_file_names = [f for f in os.listdir(test_path) if f.endswith('.tif')]
   

    valLoader = DataLoader(DatasetImageMaskContourDist_test(dir = args.test_path,file_names = test_file_names, distance_type= args.distance_type))

    if not os.path.exists(save_path_predictions):
        os.mkdir(save_path_predictions)

    model = build_model(model_type)
    # %%
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # %%
    for i, (img_file_name, inputs) in enumerate(
        tqdm(valLoader)
    ):
        print(i)
        inputs = inputs.to(device)
        outputs1, outputs2, outputs3 = model(inputs)
        print(outputs1.dtype)
        print(outputs1.shape)

        print(outputs1[0,0,1:10,1:10])
        print(outputs1[0,1,1:10,1:10])

        print(torch.max(outputs1[0,0,:,:]))
        print(torch.min(outputs1[0,0,:,:]))

        print(torch.max(outputs1[0,1,:,:]))
        print(torch.min(outputs1[0,1,:,:]))

        outputs1 = outputs1.detach().cpu().numpy().squeeze()

        #print(outputs1[0,1:10,1:10])
        #print(outputs1[1,1:10,1:10])

       
      
        # %%

        res = np.zeros((256, 256))
        # outputs1.shape()
        indices = np.argmax(outputs1, axis=0)

        print(indices[1:10,1:10])

        res[indices == 1] = 255
        res[indices == 0] = 0

        output_path = os.path.join(
            save_path_predictions, os.path.basename(img_file_name[0])
        )
        print(output_path)

        ## read orig image 
       
        orig_image_filename = f"{args.test_path}{img_file_name[0]}"
        orig_image = rio.open(orig_image_filename).read()

       
        fig, axes = plt.subplots(figsize=(8, 8), ncols= 2, nrows= 1 )
        ax, ax1 = axes.flatten()
    
        # ax.imshow(orig_image.transpose(2,1,0)[:,:,1:3])#.numpy())

      
        ax1.imshow(indices)
        
        plt.title(img_file_name)
        plt.show
    # %%
        ## TTA
        # outputs4, outputs5, outputs6 = model(torch.flip(inputs, [-1]))
        # predict_2 = torch.flip(outputs4, [-1])
        # outputs7, outputs8, outputs9 = model(torch.flip(inputs, [-2]))
        # predict_3 = torch.flip(outputs7, [-2])
        # outputs10, outputs11, outputs12 = model(torch.flip(inputs, [-1, -2]))
        # predict_4 = torch.flip(outputs10, [-1, -2])
        # predict_list = outputs1 + predict_2 + predict_3 + predict_4
        # pred1 = predict_list/4.0

        


        # %%
        
       

     
        cv2.imwrite(output_path, res)
# %%