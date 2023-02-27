# %%
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as TF

import matplotlib.pyplot as plt

#

def dice_loss(prediction, target, epoch, writer):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    print("____________________________")
    print("inside dice")

    #print(prediction[1,0,1:10,1:10])
    #print(prediction[1,1,1:10,1:10])

    #print(prediction[1,0,:,:].min())
    #print(prediction[1,0,:,:].max())
    #print(prediction[1,1,:,:].min())
    #print(prediction[1,1,:,:].max())

    ## translate into one layer
    bin_prediction = torch.ones_like(target)
    #print("prediction size: ", prediction.size())
    #print("binaere prediction size: ", bin_prediction.size())

    for t in range(prediction.size()[0]):
        #print(t)
        #bin_prediction[t,0,:,:] = torch.where(target[t,0,:,:] == 1, prediction[t,1,:,:],prediction[t,0,:,:] )
        bin_prediction[t,0,:,:] = torch.where(prediction[t,0,:,:] > 0.5, 1, 0)
    
    max_pre = torch.max(prediction)
    print(torch.max(prediction))

    # bin count
    bin_count = torch.bincount(bin_prediction.long().view(-1))
    print("ns in prediction", bin_count)
    bin_countt = torch.bincount(target.long().view(-1))
    print("ns in bin target",bin_countt)
   
    print(bin_prediction.size())
    print("bin example: ",bin_prediction.shape)
    #print("bin example: ",bin_prediction[0,0,100:110,100:110])

    i_flat = bin_prediction.flatten()
    t_flat = target.flatten()

    print("i_flat", i_flat.shape)
    print("i_flat", i_flat)
    print("t_flat", t_flat.shape)
    print("t_flat", t_flat)

    intersection = (i_flat * t_flat).sum()
    print(intersection)
    print(i_flat.sum())
    print(t_flat.sum())

    writer.add_scalar("max_pre_dice_mask", max_pre, epoch)
    writer.add_scalar("Sum_Intersection_dice_mask", intersection, epoch)
    writer.add_scalar("Loss_dice_mask", 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth)),  epoch)
    

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def dice_loss_cont(prediction, target, epoch, writer):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    print("____________________________")
    print("inside dice CONT")

    #print(prediction[1,0,1:10,1:10])
    #print(prediction[1,1,1:10,1:10])

    #print(prediction[1,0,:,:].min())
    #print(prediction[1,0,:,:].max())
    #print(prediction[1,1,:,:].min())
    #print(prediction[1,1,:,:].max())

    ## translate into one layer
    bin_prediction = torch.ones_like(target)
    #print("prediction size: ", prediction.size())
    print("binaere prediction size: ", bin_prediction.size())

    for t in range(prediction.size()[0]):
        #print(t)
       
        
        #bin_prediction[t,0,:,:] = torch.where(target[t,0,:,:] == 1, prediction[t,1,:,:],prediction[t,0,:,:] )
        bin_prediction[t,0,:,:] = torch.where(prediction[t,0,:,:] > 0.5, 1, 0)
    
    print("binaere prediction size: ", bin_prediction.size())
    # bin count
    bin_count = torch.bincount(bin_prediction.long().view(-1))
    print("ns in prediction", bin_count)
    bin_countt = torch.bincount(target.long().view(-1))
    print("ns in bin target",bin_countt)


    max_pre = torch.max(prediction)
    print(torch.max(prediction))
    print(bin_prediction.size())
    print("bin example: ",bin_prediction.shape)
    #print("bin example: ",bin_prediction[0,0,100:110,100:110])

    i_flat = bin_prediction.flatten()
    t_flat = target.flatten() # .view(-1)

    print("i_flat", i_flat.shape)
    print("i_flat", i_flat)
    print("t_flat", t_flat.shape)
    print("t_flat", t_flat)

    intersection = (i_flat * t_flat).sum()
    print("intersection", intersection)
    print("i flat sum: ", i_flat.sum())
    print("t-flat sum", t_flat.sum())
    
    writer.add_scalar("max_pre_dice_cont", max_pre, epoch)
    writer.add_scalar("Sum_Intersection_dice_cont", intersection, epoch)
    writer.add_scalar("Loss_dice_cont", 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth)),  epoch)

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def tversky(y_true, y_pred, smooth=1, alpha=0.85): #By setting the value of α > 𝜷, you can penalise false negatives more
    y_true_pos = torch.flatten(y_true)
    y_pred_pos = torch.flatten(y_pred)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, epoch, writer, gamma=0.75, alpha =0.7): # gamma:
    tv = tversky(y_true, y_pred, alpha = 0.82)
    tv = torch.pow((1 - tv), gamma)
    print("tv")
    print(tv.cpu().detach().numpy())
    print(type(tv))
    writer.add_scalar("Mask_Loss_tversky", tv.cpu().detach().numpy(), epoch)
   
    return tv

def focal_tversky_loss_cont(y_true, y_pred,  epoch, writer, gamma=0.65, alpha =0.7): # gamma:
    tv = tversky(y_true, y_pred, alpha = 0.55)
    tv = torch.pow((1 - tv), gamma)
    print("tv")
    print(tv.cpu().detach().numpy())
    print(type(tv))
    writer.add_scalar("Cont_Loss_tversky", tv.cpu().detach().numpy(), epoch)

    return tv


# ______________________________________________________________________________________________









class LossBsiNet:
    def __init__(self, weights=[1.05, 1, 1], device = torch.device("cuda")):
        #self.criterion1 = Loss(num_classes=1, device=device, class_weights = np.array([0.2,0.5]))   #mask_loss
        #self.criterion2 = Loss(num_classes=1, device=device, class_weights = np.array([0.2,0.5]))   #contour_loss
        #self.criterion1 =  log_cosh_dice_loss()   #mask_loss
        #self.criterion2 =  log_cosh_dice_loss()  
        self.criterion3 =  nn.MSELoss().to(device)               ##distance_loss
        self.weights = weights

    def __call__(self, outputs1, outputs2, outputs3, targets1, targets2, targets3, epoch, writer):
        #
        print("__________________________________________________________")
        print("inide batch loss")
        
        #fig, axes = plt.subplots(figsize=(8, 6), ncols= 3, nrows= 2 )
        #ax, ax1, ax2, ax3, ax4, ax5 = axes.flatten()

        #print(outputs1.shape)
        #print(outputs2.shape)
        #print(outputs3.shape)

        #print(targets1.shape)
        #print(targets2.shape)
        #print(targets3.shape)
        
        #outputs1_, outputs2_, outputs3_, targets1_, targets2_, targets3_ = outputs1.cpu().detach().numpy(), outputs2.cpu().detach().numpy(), outputs3.cpu().detach().numpy(), targets1.cpu().detach().numpy(), targets2.cpu().detach().numpy(), targets3.cpu().detach().numpy()

        #ax.imshow(outputs1_[-1,-1,:,:])
        #ax1.imshow(outputs2_[-1,-1,:,:])
        #ax2.imshow(outputs3_[-1,-1,:,:])

        #ax3.imshow(targets1_[-1,-1,:,:])
        #ax4.imshow(targets2_[-1,-1,:,:])
        #ax5.imshow(targets3_[-1,-1,:,:])
        ##plt.title(img_file_name)
        #plt.show"""

        weights=[1.2, 1, 1.1]


        criterion = (
                #  self.weights[0] * self.criterion1(outputs1, targets1)
                #+ self.weights[1] * self.criterion2(outputs2, targets2)
                weights[0] *    focal_tversky_loss(outputs1, targets1, epoch, writer)
                + weights[1] *  focal_tversky_loss_cont(outputs2, targets2, epoch, writer)
                + weights[2] * self.criterion3(outputs3, targets3)
        )

        return criterion





#%%