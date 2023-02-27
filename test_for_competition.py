# %%
import torch
import os
from torch.utils.data import DataLoader
from utils.data_sets import DatasetImageMaskContourDist_test
from utils.general_utils import strFilter
import glob
from models.BsiNet import BsiNet_2, BsiNet
from tqdm import tqdm
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import rasterio as rio
import pandas as pd




str_to_test_on = "2021_10"

def build_model(model_type):

    if model_type == "bsinet_2":
        model = BsiNet_2(input_channels=4 ,num_classes=2)

    return model



if __name__ == "__main__":

    args = argparse.ArgumentParser(description="test setup for segmentation test")
    
    args.add_path = "/home/hemmerling/projects/field_delination_BsiNet/"

    args.save_path_model = f"{args.add_path}models_safe/"
    args.test_path = '/data_preprocessed/test/image/'
    model_name = "bsinet_2_bathsize_10_num_epochs_200_n_args_7_ntry_11_trained_on_month_08_retrain_4_nepoch_45.pt"

    args.model_file = f"{args.add_path}models_safe/{model_name}"
 
    args.model_type = 'bsinet_2'
    args.output_folder_name = 'test_1'
    args.distance_type = 'dist_contour'
    

    args.save_path_predictions = f"{args.add_path}models_safe/predictions/{args.model_type}_{args.output_folder_name}/"
    test_path = args.test_path
    model_file = args.model_file
    save_path_model  = args.save_path_model
    save_path_predictions  = args.save_path_predictions

    model_type = args.model_type
    extra_name = "4_test"

    device = torch.device("cuda")
    test_file_names = [f for f in os.listdir(test_path) if f.endswith('.tif')]
    print(test_file_names)
    # %%

    ## filter per month 
    test_file_names = strFilter(test_file_names, ["2021_08"])

    ## read each element n list in stack
    X_test = np.empty((len(test_file_names), 4, 256, 256), dtype=np.float32)

    for u in range(len(X_test)):

        ## read
        print(test_file_names[u])
        image = rio.open(f"{args.add_path}data_preprocessed/test/image/{test_file_names[u]}").read()
        X_test[u,:,:,:] = image#.transpose(1,2,0)
  
    ## Make Predictions
    model = build_model(model_type)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    

    predictions_dictionary = {}
    for u in range(len(X_test)):
        tile_id = test_file_names[u]
        tile_id = tile_id[0:2]

        image = X_test[u,:,:,:]
        inputs = torch.tensor(image).to(device)

        outputs1, outputs2, outputs3 = model(inputs[None,:,:,:])

        outputs1 = (outputs1.cpu().detach().numpy() >= 0.5).astype(np.uint8)
        model_pred = outputs1.reshape(256, 256)
        
        predictions_dictionary.update([(str(tile_id), pd.DataFrame(model_pred))])

        fig, axes = plt.subplots(figsize=(6, 6), ncols= 2, nrows= 1 )
        ax, ax1 = axes.flatten()
      
        ax.imshow(image.transpose(1,2,0))
        ax1.imshow(model_pred)
      
        plt.show

        #print(predictions_dictionary)
    dfs = []

    for key, value in predictions_dictionary.items():
    
        ftd = value.unstack().reset_index().rename(columns={'level_0': 'row', 'level_1': 'column', 0: 'label'})
        ftd['tile_row_column'] = f'Tile{key}_' + ftd['row'].astype(str) + '_' + ftd['column'].astype(str)
        ftd = ftd[['tile_row_column', 'label']]
        dfs.append(ftd)

    sub = pd.concat(dfs)
    print(sub)

    pp = "/home/hemmerling/projects/field_delination_BsiNet/harvest_sample_submission/"
    if not os.path.isdir(pp):
        os.makedirs(pp)

    sub.to_csv(f"/home/hemmerling/projects/field_delination_BsiNet/harvest_sample_submission/{extra_name}_{model_name[:-4]}.csv", index = False)
    
    