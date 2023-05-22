#!/usr/bin/env python

import os
import sys
import glob
import tqdm
import importlib
import time
import pdb
import copy

import numpy as np
from scipy import io
import skimage.io as sio
from scipy import ndimage
import cv2

import torch
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt
plt.gray()

from modules import models
from modules import utils
from modules import volutils
import tifffile


def get_depth_mask(D, H, W, slices):
    # Generate a tensor of the depth index for each element in the flattened tensors
    depth_indices = torch.arange(D * H * W) // (H * W)
    depth_indices = depth_indices.cuda()

    # Generate a boolean mask indicating whether each element is in one of the desired slices
    mask = torch.isin(depth_indices, torch.tensor(slices, device='cuda'))

    return mask

def get_width_mask(D, H, W, slices):
    # Generate a tensor of the width index for each element in the flattened tensors
    width_indices = torch.arange(D * H * W) % W
    width_indices = width_indices.cuda()

    # Generate a boolean mask indicating whether each element is in one of the desired slices
    mask = torch.isin(width_indices, torch.tensor(slices, device='cuda'))

    return mask

if __name__ == '__main__':
    nonlin = 'relu' # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = 4000                # Number of SGD iterations
    learning_rate = 5e-3        # Learning rate
    expname = '5_SPIMA_noAffine'     # Volume to load
    expname2 = '5_SPIMB_noAffine'  # Volume to load
    scale = 1                 # Run at lower scales to testing
    mcubes_thres = 0.5          # Threshold for marching cubes
    
    # Gabor filter constants
    # These settings work best for 3D occupancies
    omega0 = 10.0          # Frequency of sinusoid
    sigma0 = 40.0          # Sigma of Gaussian
    
    # Network constants
    hidden_layers = 2       # Number of hidden layers in the mlp
    hidden_features = 256   # Number of hidden units per layer
    maxpoints = int(2e5)    # Batch size
    
    if expname == 'thai_statue':
        occupancy = True
    else:
        occupancy = False
    
    # Load image and scale
    # im = io.loadmat('data/%s.mat'%expname)['hypercube'].astype(np.float32)
    # im = ndimage.zoom(im/im.max(), [scale, scale, scale], order=0)

    im = tifffile.imread(f'data/{expname}.tiff')
    im = np.float32(im)

    im2 = tifffile.imread(f'data/{expname2}.tiff')
    im2 = np.float32(im2)
    
    # If the volume is an occupancy, clip to tightest bounding box
    if occupancy:
        hidx, widx, tidx = np.where(im > 0.99)
        im = im[hidx.min():hidx.max(),
                widx.min():widx.max(),
                tidx.min():tidx.max()]
    
    print(im.shape)
    D, H, W = im.shape
    
    maxpoints = min(D * H * W, maxpoints)

    # Get the mask for the desired slices
    slices = torch.arange(0, D, 2).cuda()  # 0, 10, 20, ..., 110
    maskD = get_depth_mask(D, H, W, slices)
    maskW = get_width_mask(D, H, W, slices)

    imten = torch.tensor(im).cuda().reshape(D * H * W, 1)
    imten = imten[maskD]

    imten2 = torch.tensor(im2).cuda().reshape(D * H * W, 1)
    imten2 = imten2[maskW]
    
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
    else:
        posencode = False
    
    # Create model
    model = models.get_INR(
                    nonlin=nonlin,
                    in_features=3,
                    out_features=1, 
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    first_omega_0=omega0,
                    hidden_omega_0=omega0,
                    scale=sigma0,
                    pos_encode=posencode,
                    sidelength=max(D, H, W)).cuda()
    
    # Optimizer
    optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    
    # Schedule to 0.1 times the initial rate
    scheduler = LambdaLR(optim, lambda x: 0.2**min(x/niters, 1))

    criterion = torch.nn.MSELoss()
    
    # Create inputs
    coords = utils.get_coords(D, H, W)
    coords = coords.cuda()
    coordsD = coords[maskD]
    coordsW = coords[maskW]

    maxpointsD = min(len(coordsD), maxpoints)
    maxpointsW = min(len(coordsW), maxpoints)
    
    mse_array = np.zeros(niters)
    time_array = np.zeros(niters)
    best_mse = float('inf')
    best_img = None

    tbar = tqdm.tqdm(range(niters))
    
    im_estim = torch.zeros(coordsD.shape[0], 1, device='cuda')
    im_estim2 = torch.zeros(coordsW.shape[0], 1, device='cuda')
    
    tic = time.time()
    print('Running %s nonlinearity'%nonlin)
    # for idx in tbar:
    #     indicesD = torch.randperm(len(coordsD))
    #     indicesW = torch.randperm(len(coordsW))
    #
    #     train_loss = 0
    #     nchunks = 0
    #     for b_idx in range(0, max(len(coordsD), len(coordsW)), maxpointsD):
    #         # Get batch indices for D and W
    #         b_indicesD = indicesD[b_idx:min(len(coordsD), b_idx + maxpointsD)]
    #         b_indicesW = indicesW[b_idx:min(len(coordsW), b_idx + maxpointsD)]
    #
    #         # Get batch coordinates for D and W
    #         b_coordsD = coordsD[b_indicesD, ...].cuda()
    #         b_coordsW = coordsW[b_indicesW, ...].cuda()
    #
    #         # Concatenate batch coordinates for D and W
    #         b_coords = torch.cat([b_coordsD, b_coordsW], dim=0)
    #
    #         # Get batch pixelvalues for D and W
    #         pixelvaluesD = imten[b_indicesD, :]
    #         pixelvaluesW = imten2[b_indicesW, :]
    #
    #         # Concatenate batch pixelvalues for D and W
    #         pixelvalues_true = torch.cat([pixelvaluesD, pixelvaluesW], dim=0).cuda()
    #
    #         # Forward pass
    #         pixelvalues_pred = model(b_coords[None, ...]).squeeze()[:, None]
    #
    #         # Compute loss
    #         loss = criterion(pixelvalues_pred, pixelvalues_true)
    #
    #         # Backward pass and optimization
    #         optim.zero_grad()
    #         loss.backward()
    #         optim.step()
    #
    #         # Update train loss
    #         lossval = loss.item()
    #         train_loss += lossval
    #         nchunks += 1



    for idx in tbar:
        indicesD = torch.randperm(len(coordsD))
        indicesW = torch.randperm(len(coordsW))
        
        train_loss = 0
        nchunks = 0
        for b_idx in range(0, len(coordsD), maxpointsD):
            b_indices = indicesD[b_idx:min(len(coordsD), b_idx + maxpointsD)]
            b_coords = coordsD[b_indices, ...].cuda()
            b_indices = b_indices.cuda()
            pixelvalues = model(b_coords[None, ...]).squeeze()[:, None]

            loss = criterion(pixelvalues, imten[b_indices, :])
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            lossval = loss.item()
            train_loss += lossval
            nchunks += 1

        for b_idx in range(0, len(coordsW), maxpointsW):
            b_indices = indicesW[b_idx:min(len(coordsW), b_idx + maxpointsW)]
            b_coords = coordsW[b_indices, ...].cuda()
            b_indices = b_indices.cuda()
            pixelvalues = model(b_coords[None, ...]).squeeze()[:, None]

            loss = criterion(pixelvalues, imten2[b_indices, :])

            optim.zero_grad()
            loss.backward()
            optim.step()

            lossval = loss.item()
            train_loss += lossval
            nchunks += 1

        if occupancy:
            mse_array[idx] = volutils.get_IoU(im_estim, imten, mcubes_thres)
        else:
            mse_array[idx] = train_loss/nchunks
        time_array[idx] = time.time()
        scheduler.step()
        
        if lossval < best_mse:
            best_mse = lossval
            best_img = copy.deepcopy(im_estim)

        tbar.set_description('%.4e'%mse_array[idx])
        tbar.refresh()
        
    total_time = time.time() - tic
    nparams = utils.count_parameters(model)

    coords = utils.get_coords(D, H, W)
    im_estim = torch.zeros(coords.shape[0], 1, device='cuda')
    indices = torch.randperm(len(coords))
    with torch.no_grad():
        for b_idx in range(0, len(coords), maxpoints):
            b_indices = indices[b_idx:min(len(coords), b_idx + maxpoints)]
            b_coords = coords[b_indices, ...].cuda()
            b_indices = b_indices.cuda()
            pixelvalues = model(b_coords[None, ...]).squeeze()[:, None]

            im_estim[b_indices, :] = pixelvalues

    best_img = im_estim.reshape(D, H, W).detach().cpu().numpy()
    
    if posencode:
        nonlin = 'posenc'
        
    # Save data
    os.makedirs('results/%s'%expname, exist_ok=True)
    
    indices, = np.where(time_array > 0)
    time_array = time_array[indices]
    mse_array = mse_array[indices]
    
    mdict = {'mse_array': mse_array,
             'time_array': time_array-time_array[0],
             'nparams': utils.count_parameters(model)}
    io.savemat('results/%s/%s.mat'%(expname, nonlin), mdict)
    
    # Generate a mesh with marching cubes if it is an occupancy volume
    if occupancy:
        savename = 'results/%s/%s.dae'%(expname, nonlin)
        volutils.march_and_save(best_img, mcubes_thres, savename, True)
    
    print('Total time %.2f minutes'%(total_time/60))
    if occupancy:
        print('IoU: ', volutils.get_IoU(best_img, im, mcubes_thres))
    else:
        print('PSNR: ', utils.psnr(im, best_img))
    print('Total pararmeters: %.2f million'%(nparams/1e6))


    # Save the array as a TIFF file
    sio.imsave(f'results/{expname}/output_dual.tiff', best_img)

    
    
