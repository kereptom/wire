#!/usr/bin/env python

import os
import tqdm
import time
import copy
import numpy as np
from scipy import io
import skimage.io as sio
import torch
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
plt.gray()
from modules import models
from modules import utils
from modules import volutils
from PIL import Image
import tifffile
import wandb


def process_image(img):
    # Compute the MIPs along three axes
    mip_xy = np.max(img, axis=0)
    mip_xz = np.max(img, axis=1)
    mip_yz = np.max(img, axis=2)

    # Convert arrays into images
    im_mip_xy = Image.fromarray(mip_xy)
    im_mip_xz = Image.fromarray(mip_xz)
    im_mip_yz = Image.fromarray(mip_yz)

    # Concatenate along the horizontal axis (change axis as required)
    concat = Image.new('L', (3 * im_mip_xy.width, im_mip_xy.height))
    concat.paste(im_mip_xy, (0, 0))
    concat.paste(im_mip_xz, (im_mip_xy.width, 0))
    concat.paste(im_mip_yz, (2 * im_mip_xy.width, 0))

    return concat


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

    # Wandb logging
    run = wandb.init(project='wire')

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

    verbose_freq = niters//10
    
    # Network constants
    hidden_layers = 2       # Number of hidden layers in the mlp
    hidden_features = 256   # Number of hidden units per layer
    maxpoints = int(2e5)    # Batch size

    im = tifffile.imread(f'data/{expname}.tiff')
    im = np.float32(im)

    im2 = tifffile.imread(f'data/{expname2}.tiff')
    im2 = np.float32(im2)

    
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

        if not idx % verbose_freq or idx == niters - 1:
            # Process images and upload to wandb
            concat1 = process_image(best_img)
            concat2 = process_image(im)
            concat3 = process_image(im2)

            wandb.log({
                "concat1": wandb.Image(concat1),
                "concat2": wandb.Image(concat2),
                "concat3": wandb.Image(concat3)
            })

        mse_array[idx] = train_loss/nchunks
        time_array[idx] = time.time()

        wandb.log({'mse': mse_array[idx], 'time': time_array[idx]}, step=idx)

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

    print('Total time %.2f minutes'%(total_time/60))
    print('PSNR: ', utils.psnr(im, best_img))
    print('Total pararmeters: %.2f million'%(nparams/1e6))

    sio.imsave(f'results/{expname}/output_dual1.tiff', best_img)

    run.finish()