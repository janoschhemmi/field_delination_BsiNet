# %%
import argparse
import sys
import logging
from tqdm import tqdm
import glob 
import os
import rasterio as rio
from sklearn.model_selection import train_test_split
import torch 
from torch import nn
import numpy as np 
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.losses import LossBsiNet
from utils.general_utils import visualize, create_train_arg_parser,evaluate, strFilter
from models.BsiNet import BsiNet_2, BsiNet
import matplotlib.pyplot as plt
from torch.nn import functional as F

from utils.data_sets import readTif, DatasetImageMaskContourDist

# %%

def define_loss(loss_type, weights=[1.5, 1, 1], device = torch.device("cuda")):

    if loss_type == "bsinet":
        criterion = LossBsiNet(weights, device=device)

    return criterion

def build_model(model_type):

    if model_type == "bsinet":
     
        model = BsiNet(num_classes=2)

    if model_type == "bsinet_2":

        model = BsiNet_2(input_channels = 4, num_classes=2)

    return model


def train_model(model, inputs, targets, model_type, criterion, optimizer, device, epoch, writer):

    if model_type == "bsinet_2":

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            inputs = inputs.to(device).type(torch.cuda.FloatTensor)
      
            outputs = model(inputs)

            loss = criterion(
                outputs[0], outputs[1], outputs[2], targets[0], targets[1], targets[2], epoch, writer
            )
            loss.backward()
            optimizer.step()

    return loss



#%%

if __name__ == "__main__":

    ## set cuda max memory
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print("total cuda memory:    ",total_memory / 1e-6)
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    application = int(total_memory * 0.2) - torch.cuda.max_memory_reserved()
    print("memory alloed to use: ", application)

    
    ## GLobals
    args = argparse.ArgumentParser(description="train setup for segmentation")
    args.test_path = "path to test"
    args.model_file = "file "
    args.save_path = "path "
    args.cuda_no = "no"

    ## optional load of pretrained models from disk 
    args.use_pretrained = False
    ## optinal sanity check ofInput data
    args.sanity = False

    args.train_path = '/home/hemmerling/projects/field_delination_BsiNet/data_preprocessed_2/train/image'
    args.save_path = '/home/hemmerling/projects/field_delination_BsiNet/models_safe'
    args.object_type = "duno"

    ## Model params
    args.model_type = "bsinet_2"
    args.batch_size = 10
    args.val_batch_size = 8
    args.distance_type = 'dist_contour'

    ## Training params
    args.num_epochs = 200
    args.ntry = 11
    args.n_args = 7
    args.trained_on_month_ = "08_retrain_4"
    args.safe_model_from_epoch = 30 

    ## create model run name 
    model_run_name = f"{args.model_type}_bathsize_{args.batch_size}_num_epochs_{args.num_epochs}_n_args_{args.n_args}_ntry_{args.ntry}_trained_on_month_{args.trained_on_month_}"
    
    args.pretrained_model_path = "/home/hemmerling/projects/field_delination_BsiNet/models_safe/bsinet_2_bathsize_20_num_epochs_200_n_args_7_ntry_10_trained_on_month_04_retrain_3_nepoch_105.pt"
    log_path = f"{args.save_path}/runs/{model_run_name}/"

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_dir=log_path)

    train_file_names = glob.glob(os.path.join(args.train_path, "*.tif"), recursive=False)

    ## filter 
    train_file_names = strFilter(train_file_names, ["2021_04","2021_08","2021_10","a60","a61","a62" ,"a63","a64","a65","a66","05norm"])
    train_file_names = strFilter(train_file_names, ["a60","a61","a62","a63","a64","a65","a66"])
    train_file_names = strFilter(train_file_names, ["2021_08"])
    train_file_names = strFilter(train_file_names, ["05norm"])

    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_file_names]
    train_file, val_file = train_test_split(img_ids, test_size=0.1, random_state=41)

    ## set cuda device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda")
    #device = torch.device("cpu")
    print("Device used: ", device)

    ## load model 
    model = build_model(args.model_type)
    model = model.to(device)
  
    ### Load pretrained model if stated 
    epoch_start = "345"
    if args.use_pretrained:
        print("Loading Model {}".format(os.path.basename(args.pretrained_model_path)))
        model.load_state_dict(torch.load(args.pretrained_model_path))            #加了False
 

    trainLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path, train_file, args.distance_type),
        batch_size=args.batch_size,drop_last=True,  shuffle=True
        )

    valLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path,val_file, args.distance_type),drop_last=True,
    )
 
    displayLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path,val_file, args.distance_type),
        batch_size=args.val_batch_size,drop_last=True, shuffle=True
    )
     

    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e10), eta_min=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 0.1) 
    
    ## get Loss function 
    criterion = define_loss("bsinet", device= device)

    ## ___________________________________________________________________________________

    for epoch in tqdm(range( int(epoch_start) + 1 + args.num_epochs)):

        global_step = epoch * len(trainLoader)
        running_loss = 0.0

        for i, (img_file_name, inputs, targets1, targets2,targets3) in enumerate(
            tqdm(trainLoader)
        ):

            model.train()
            targets1 = targets1[:,-1,:,:,:]
            targets2 = targets2[:,-1,:,:,:]

            ## sanity check 
            if args.sanity == True:
                
                ## get first 
                print("___________________________________________________")
                print("sanity check")

                print("input")
                sanity_input = inputs[0]
                print(sanity_input.shape)
                print(sanity_input.dtype)
                print(sanity_input.min())
                print(sanity_input.max())
                print("tar 1")
                sanity_target_1 = targets1[0]
                print(sanity_target_1.shape)
                print(sanity_target_1.dtype)
                print(sanity_target_1.min())
                print(sanity_target_1.max())
                print("tar 2")
                sanity_target_2 = targets2[0]
                print(sanity_target_2.shape)
                print(sanity_target_2.dtype)
                print(sanity_target_2.min())
                print(sanity_target_2.max())
                print("tar 3")
                sanity_target_3 = targets3[0]
                print(sanity_target_3.shape)
                print(sanity_target_3.dtype)
                print(sanity_target_3.min())
                print(sanity_target_3.max())

                fig, axes = plt.subplots(figsize=(8, 8), ncols= 4, nrows= 1 )
                ax, ax1, ax2, ax3 = axes.flatten()
   
                ax.imshow(sanity_input.permute(2,1,0)[:,:,0:3])
                ax1.imshow(sanity_target_1.permute(2,1,0))
                ax2.imshow(sanity_target_2.permute(2,1,0))
                ax3.imshow(sanity_target_3.permute(2,1,0))
               
                plt.title(img_file_name)
                plt.show
                

            inputs   = inputs.to(device,  dtype=torch.float)
            targets1 = targets1.to(device,  dtype=torch.float)
            targets2 = targets2.to(device,  dtype=torch.float)
            targets3 = targets3.to(device,  dtype=torch.float)

            targets = [targets1, targets2,targets3]
    
            loss = train_model(model, inputs, targets, args.model_type, criterion, optimizer, device, epoch, writer)
            writer.add_scalar("batch loss", loss.item(), epoch)

            running_loss += loss.item() * inputs.size(0)
              
        scheduler.step()

        epoch_loss = running_loss / len(train_file_names)
        print("Loss of epoch: ", epoch_loss)
    
        
        if epoch % 1 == 0:

            val_loss, dev_time = evaluate(device, epoch, model, valLoader, writer)
            writer.add_scalar("loss_valid", val_loss, epoch)
            visualize(device, epoch, model, displayLoader, writer, args.val_batch_size)
            print("Global Loss:{} Val Loss:{}".format(epoch_loss, val_loss))
        else:
            print("Global Loss:{} ".format(epoch_loss))

        logging.info("epoch:{} train_loss:{} ".format(epoch, epoch_loss))
        if epoch % 15 == 0:

            if epoch > args.safe_model_from_epoch:
                torch.save(
                    model.state_dict(), f"{args.save_path}/{model_run_name}_nepoch_{epoch}.pt")
            


# %%

##print  distance cont and mask 
## chec if all are the same 

