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
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.losses import LossBsiNet
from utils.general_utils import visualize, create_train_arg_parser,evaluate
from models.BsiNet import BsiNet_2, BsiNet

#from utils.data_sets import DatasetImageMaskContourDist
from utils.data_sets import readTif, DatasetImageMaskContourDist




def define_loss(loss_type, weights=[1, 1, 1]):

    if loss_type == "bsinet":
        criterion = LossBsiNet(weights) 

    return criterion

def build_model(model_type):

    if model_type == "bsinet":
        model = BsiNet(num_classes=2)

    if model_type == "bsinet_2":
        model = BsiNet_2(input_channels = 4, num_classes=2)

    return model


def train_model(model, targets, model_type, criterion, optimizer):

    if model_type == "bsinet_2":

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(
                outputs[0], outputs[1], outputs[2], targets[0], targets[1], targets[2]
            )
            loss.backward()
            optimizer.step()

    return loss

## set cuda max memory
torch.cuda.empty_cache()
total_memory = torch.cuda.get_device_properties(0).total_memory
print("total cuda memory:    ",total_memory)
torch.cuda.set_per_process_memory_fraction(0.2, 0)
application = int(total_memory * 0.2) - torch.cuda.max_memory_reserved()
print("memory alloed to use: ", application)

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

    args.test_path = "path to test"
    args.model_file = "file "
    args.save_path = "path "
    args.cuda_no = "no"


    ## optional load of pretrained models from disk 
    args.use_pretrained = False
    # args.pretrained_model_path = './best_merge_model_article/85.pt'


    args.train_path = '/home/hemmerling/projects/field_delination_BsiNet/data_preprocessed/train/image'
    args.val_path = '/home/hemmerling/projects/field_delination_BsiNet/data'
    args.save_path = './model'
    args.object_type = "duno"

    ## Model params
    args.model_type = "bsinet_2"
    args.batch_size = 2
    args.val_batch_size = 2
    args.distance_type = 'dist_contour'

    ## Training params
    args.num_epochs = 3



    train_file_names = glob.glob(os.path.join(args.train_path, "*.tif"), recursive=False)
    #val_file_names = glob.glob(os.path.join(args.val_path, "*.tif"))

    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_file_names]
    train_file, val_file = train_test_split(img_ids, test_size=0.2, random_state=41)

    ## set cuda device 
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    print(device)

    ## load model 
    model = build_model(args.model_type)
    """ if torch.cuda.device_count() > 0:           #本来是0
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)"""
    model = model.to(device)

    ### Load pretrained model if stated 
    epoch_start = "0"
    if args.use_pretrained:
        print("Loading Model {}".format(os.path.basename(args.pretrained_model_path)))
        model.load_state_dict(torch.load(args.pretrained_model_path))            #加了False
        epoch_start = os.path.basename(args.pretrained_model_path).split(".")[0]
        print(epoch_start)


    # print(args.train_path, train_file, args.distance_type)
    """from utils.data_sets import load_image, load_mask, load_contour, load_distance
    print(os.path.join(args.train_path, train_file[1]+'.tif'))
    from PIL import Image
    pp = rio.open(os.path.join(args.train_path, train_file[1]+'.tif'))
    print(pp.shape)
    pp = load_image(os.path.join(args.train_path, train_file[1]+'.tif'))
    print(pp.shape)

    pp = load_mask(os.path.join(args.train_path, train_file[1]+'.tif'))
    print(pp.shape)

    pp = load_contour(os.path.join(args.train_path, train_file[1]+'.tif'))
    print(pp.shape)

    pp = load_distance(os.path.join(args.train_path, train_file[1]+'.tif'), "dist_contour")
    print(pp.shape)"""

    # %%
    trainLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path, train_file, args.distance_type),
        batch_size=args.batch_size,drop_last=False,  shuffle=True
        )

    devLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path,val_file, args.distance_type),drop_last=True,
    )
    displayLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path,val_file, args.distance_type),
        batch_size=args.val_batch_size,drop_last=True, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e10), eta_min=1e-5)
   # scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 0.1) 
    criterion = define_loss("bsinet")

    for epoch in tqdm(range( int(epoch_start) + 1 + args.num_epochs)):

        global_step = epoch * len(trainLoader)
        running_loss = 0.0

        print(global_step)

        for i, (img_file_name, inputs, targets1, targets2,targets3) in enumerate(
            tqdm(trainLoader)
        ):

            model.train()

            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            targets3 = targets3.to(device)

            targets = [targets1, targets2,targets3]


            loss = train_model(model, targets, args.model_type, criterion, optimizer)

            writer.add_scalar("loss", loss.item(), epoch)

            running_loss += loss.item() * inputs.size(0)
        scheduler.step()

        epoch_loss = running_loss / len(train_file_names)
        print(epoch_loss)

# %%
