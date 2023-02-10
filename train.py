# %%
import argparse
import sys
import logging
import glob 
import os
from sklearn.model_selection import train_test_split
import torch 
from torch import nn

from utils.losses import LossBsiNet
from utils.general_utils import visualize, create_train_arg_parser,evaluate
from models.BsiNet import BsiNet_2, BsiNet
#from utils.data_sets import DatasetImageMaskContourDist
from utils.data_sets import readTif





def define_loss(loss_type, weights=[1, 1, 1]):

    if loss_type == "bsinet":
        criterion = LossBsiNet(weights) 

    return 

def build_model(model_type):

    if model_type == "bsinet":
        model = BsiNet(num_classes=2)

    if model_type == "bsinet_2":
        model = BsiNet_2(input_channels = 4, num_classes=2)

    return model


def train_model(model, targets, model_type, criterion, optimizer):

    if model_type == "bsinet":

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(
                outputs[0], outputs[1], outputs[2], targets[0], targets[1], targets[2]
            )
            loss.backward()
            optimizer.step()

    return loss


#%%

if __name__ == "__main__":

    #args = create_train_arg_parser().parse_args()

    ## GLobals
    #model_type = "bsinet"
    #test_path  =
    #model_file = 
    #save_path  =
    #cuda_no    = 

    args = argparse.ArgumentParser(description="train setup for segmentation")

    args.model_type = "bsinet_2"
    args.test_path = "path to test"
    args.model_file = "file "
    args.save_path = "path "
    args.cuda_no = "no"

    ## optional load of pretrained models from disk 
    args.use_pretrained = False
    # args.pretrained_model_path = './best_merge_model_article/85.pt'

    
    

    args.train_path = '/home/hemmerling/projects/field_delination/augmented_data/source_train/source_train_00/'
    args.val_path = '/home/hemmerling/projects/field_delination/augmented_data/source_train/source_train_00/'
    args.model_type = 'bsinet'
    args.save_path = './model'
    args.object_type = "duno"


    train_file_names = glob.glob(os.path.join(args.train_path, "*.tif"), recursive=False)
    val_file_names = glob.glob(os.path.join(args.val_path, "*.tif"))

    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_file_names]
    print(img_ids)
    train_file, val_file = train_test_split(img_ids, test_size=0.2, random_state=41)

    ## set cuda device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ## load model 
    model = build_model(args.model_type)
    if torch.cuda.device_count() > 0:           #本来是0
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    ### Load pretrained model if stated 
    epoch_start = "0"
    if args.use_pretrained:
        print("Loading Model {}".format(os.path.basename(args.pretrained_model_path)))
        model.load_state_dict(torch.load(args.pretrained_model_path))            #加了False
        epoch_start = os.path.basename(args.pretrained_model_path).split(".")[0]
        print(epoch_start)


    #trainLoader = DataLoader(
    #    DatasetImageMaskContourDist(args.train_path,train_file, args.distance_type),
    #    batch_size=args.batch_size,drop_last=False,  shuffle=True
    #    )



# %%
