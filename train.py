# %%
import argparse
import sys
import logging
import glob 
import os
from sklearn.model_selection import train_test_split
import torch 

from utils.losses import LossBsiNet
from utils.general_utils import visualize, create_train_arg_parser,evaluate
from models.BsiNe



def define_loss(loss_type, weights=[1, 1, 1]):

    if loss_type == "bsinet":
        criterion = LossBsiNet(weights) 

    return 

def build_model(model_type):

    if model_type == "bsinet":
        model = BsiNet(num_classes=2)

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

    args.model_type = "bsinet"
    args.test_path = "path to test"
    args.model_file = "file "
    args.save_path = "path "
    args.cuda_no = "no"

    
    

    args.train_path = '/home/hemmerling/projects/field_delination/augmented_data/source_train/source_train_00/'
    args.val_path = '/home/hemmerling/projects/field_delination/augmented_data/source_train/source_train_00/'
    args.model_type = 'bsinet'
    args.save_path = './model'
    args.object_type = "duno"

   
    print(args.train_path)

    """    
    logging.basicConfig(
        filename="".format(args.object_type),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    logging.info("")"""


    train_file_names = glob.glob(os.path.join(args.train_path, "*.tif"), recursive=False)
    val_file_names = glob.glob(os.path.join(args.val_path, "*.tif"))

    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_file_names]
    print(img_ids)
    train_file, val_file = train_test_split(img_ids, test_size=0.2, random_state=41)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = build_model(args.model_type)

# %%
