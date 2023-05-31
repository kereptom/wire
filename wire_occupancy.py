import tqdm
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from modules import models
from modules import utils
from PIL import Image
import tifffile
import wandb
import random
from PIL import ImageDraw, ImageFont
import nltk
from nltk.corpus import words
import skimage
from skimage import io
import os


def prepare_hyperparameters():
    return dict(
        project='wire',
        nonlin='wire',
        verbose=True,
        verbose_freq=100,
        niters=10000,
        cycle_start=3000,
        lr=0.0014606325337710355,
        original='3_GT',
        expname='5_SPIMA_noAffine',
        expname2='5_SPIMB_noAffine',
        psfA='psfA',
        psfB='psfB',
        bit_depth=16,
        scale=1,
        omega0=3.0,
        sigma0=1.0,
        hidden_layers=2,
        hidden_features=128,
        in_features=3,
        out_features=1,
        maxpoints=int(2e5),
        device='cuda'
    )


def normalize(img_np, invert=False):
    img_min = np.min(img_np)
    img_max = np.max(img_np)

    new_min = 0
    new_max = 1
    img_normd = (img_np - img_min) * ((new_max - new_min) / (img_max - img_min)) + new_min

    if invert:
        img_normd = new_max - img_normd

    return img_normd


def set_seeds(seed=7):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def get_random_word():
    return random.choice(words.words())


def init_wandb(hyperparameters):
    run = wandb.init(config=hyperparameters, project=hyperparameters['project'])
    run_id = run.id
    return wandb.config, run, run_id


def process_image(img):
    mip_images = [Image.fromarray(np.asarray((np.max(img, axis=i) + 1) * 127).astype(np.uint8)) for i in range(3)]
    captions = ['XY MIP', 'XZ MIP', 'YZ MIP']

    padding = 50  # Adjust this value as needed
    # Increase canvas size to accommodate padding
    concat = Image.new('L', (3 * mip_images[0].width, mip_images[0].height + padding))
    draw = ImageDraw.Draw(concat)
    font = ImageFont.load_default()  # use a default font

    for i, mip in enumerate(mip_images):
        # Paste images lower to accommodate padding
        concat.paste(mip, (i * mip.width, padding))

        # Calculate the bounding box of the text to be drawn
        text_bbox = draw.textbbox((0, 0), captions[i], font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (i * mip.width) + ((mip.width - text_width) // 2)  # Centered text
        text_y = (padding // 2) - (text_height // 2)  # Centered in padding

        draw.text((text_x, text_y), captions[i], font=font, fill=255)

    return concat


def get_depth_mask(D, H, W, slices):
    # Generate a tensor of the depth index for each element in the flattened tensors
    depth_indices = torch.arange(D * H * W) // (H * W)
    depth_indices = depth_indices.to(config.device)

    # Generate a boolean mask indicating whether each element is in one of the desired slices
    mask = torch.isin(depth_indices, slices.clone().detach())

    return mask


def get_width_mask(D, H, W, slices):
    # Generate a tensor of the width index for each element in the flattened tensors
    width_indices = torch.arange(D * H * W) % W
    width_indices = width_indices.to(config.device)

    # Generate a boolean mask indicating whether each element is in one of the desired slices
    mask = torch.isin(width_indices, slices.clone().detach())

    return mask


def imProcessRGB_3D(im1, bit_depth=16):
    max_val = 2 ** bit_depth - 1
    imNorm = lambda x: (x - 0.5) * 2
    im1 = torch.FloatTensor(im1.astype(np.float32) * (1.0 / max_val)).to(config.device).reshape(D * H * W, 1)
    return imNorm(im1)


def lr_lambda(x):
    if x < 1000:
        return 1.0
    elif 1000 <= x < 2000:
        return 0.5
    else:
        return 2


if __name__ == '__main__':

    set_seeds()
    hyperparameters = prepare_hyperparameters()
    config, run, run_id = init_wandb(hyperparameters)

    gt = np.float32(tifffile.imread(f'data/{config.original}.tiff'))
    gt_normalized = normalize(gt.astype(np.float32))
    im = np.float32(tifffile.imread(f'data/{config.expname}.tiff'))
    im_normalized = normalize(im.astype(np.float32))
    im2 = np.float32(tifffile.imread(f'data/{config.expname2}.tiff'))
    im2_normalized = normalize(im2.astype(np.float32))

    gau_filterA = np.float32(tifffile.imread(f'data/{config.psfA}.tiff'))
    gau_filterB = np.float32(tifffile.imread(f'data/{config.psfB}.tiff'))
    gau_filterA_T = gau_filterA[::-1, ::-1, ::-1]
    gau_filterB_T = gau_filterB[::-1, ::-1, ::-1]
    cube_filterA = torch.Tensor(gau_filterA_T.copy()).unsqueeze(0).unsqueeze(0).to(config.device)
    cube_filterB = torch.Tensor(gau_filterB_T.copy()).unsqueeze(0).unsqueeze(0).to(config.device)
    cube_filterA = cube_filterA / cube_filterA.sum()
    cube_filterB = cube_filterB / cube_filterB.sum()

    print(f'[INFO] Image shape: {im.shape}')
    D, H, W = im.shape
    maxpts = min(D * H * W, config.maxpoints)

    # Get the mask for the desired slices
    slices = torch.arange(0, D, 1).to(config.device)  # 0, 10, 20, ..., 110
    maskD = get_depth_mask(D, H, W, slices)
    maskW = get_width_mask(D, H, W, slices)

    im_ten = imProcessRGB_3D(im, bit_depth=config.bit_depth)
    imten = im_ten[maskD]

    im_ten2 = imProcessRGB_3D(im2, bit_depth=config.bit_depth)
    imten2 = im_ten2[maskW]

    if config.nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
    else:
        posencode = False

    # Create model
    model = models.get_INR(
        nonlin=config.nonlin,
        in_features=config.in_features,
        out_features=config.out_features,
        hidden_features=config.hidden_features,
        hidden_layers=config.hidden_layers,
        first_omega_0=config.omega0,
        hidden_omega_0=config.omega0,
        scale=config.sigma0,
        pos_encode=posencode,
        sidelength=max(D, H, W)).to(config.device)

    # Optimizer
    optim = torch.optim.Adam(lr=config.lr, params=model.parameters())

    # Schedule to 0.1 times the initial rate
    scheduler = LambdaLR(optim, lambda x: 0.2 ** min(x / config.niters, 1))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    criterion = torch.nn.MSELoss()

    # Create inputs
    coords = utils.get_coords(D, H, W)
    coords = coords.to(config.device)
    coordsD = coords[maskD]
    coordsW = coords[maskW]

    maxpointsD = min(len(coordsD), maxpts)
    maxpointsW = min(len(coordsW), maxpts)

    mse_array = np.zeros(config.niters)
    time_array = np.zeros(config.niters)
    tbar = tqdm.tqdm(range(config.niters))

    im_estim = torch.zeros(coordsD.shape[0], 1, device='cuda')
    im_estim2 = torch.zeros(coordsW.shape[0], 1, device='cuda')

    tic = time.time()
    print('Running %s nonlinearity' % config.nonlin)
    for idx in tbar:

        if idx < config.cycle_start:
            indicesD = torch.randperm(len(coordsD))
            indicesW = torch.randperm(len(coordsW))

            train_loss = 0
            nchunks = 0
            for b_idx in range(0, len(coordsD), maxpointsD):
                b_indices = indicesD[b_idx:min(len(coordsD), b_idx + maxpointsD)]
                b_coords = coordsD[b_indices, ...].to(config.device)
                b_indices = b_indices.to(config.device)
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
                b_coords = coordsW[b_indices, ...].to(config.device)
                b_indices = b_indices.to(config.device)
                pixelvalues = model(b_coords[None, ...]).squeeze()[:, None]

                loss = criterion(pixelvalues, imten2[b_indices, :])

                optim.zero_grad()
                loss.backward()
                optim.step()

                lossval = loss.item()
                train_loss += lossval
                nchunks += 1

            mse_array[idx] = train_loss / nchunks
            time_array[idx] = time.time()

            wandb.log({'mse': mse_array[idx]}, step=idx)

            scheduler.step()
        else:
            coords = utils.get_coords(D, H, W)
            im_sharp = torch.zeros(coords.shape[0], 1, device='cuda')
            for b_idx in range(0, len(coords), maxpts):
                b_indices = indices[b_idx:min(len(coords), b_idx + maxpts)]
                b_coords = coords[b_indices, ...].to(config.device)
                b_indices = b_indices.to(config.device)
                pixelvalues = model(b_coords[None, ...]).squeeze()[:, None]
                im_sharp[b_indices, :] = pixelvalues
            im_sharp = im_sharp.reshape(D, H, W).unsqueeze(0).unsqueeze(0)

            # bluredA = torch.nn.functional.conv3d(im_sharp, cube_filterA, padding='same')
            # bluredB = torch.nn.functional.conv3d(im_sharp, cube_filterB, padding='same')

            padding_size = cube_filterA.shape[2] // 2  # assuming cube_filterA is of shape (F, F, F)
            padding = (padding_size, padding_size, padding_size, padding_size, padding_size, padding_size)

            # Now pad the input
            im_sharp_pad = torch.nn.functional.pad(im_sharp, padding, mode='constant', value=-1)

            # Now use 'valid' padding (i.e., no padding) in the conv3d operation
            bluredA = torch.nn.functional.conv3d(im_sharp_pad, cube_filterA)
            bluredB = torch.nn.functional.conv3d(im_sharp_pad, cube_filterB)

            loss1 = criterion(bluredA, imten.reshape(D, H, W).unsqueeze(0).unsqueeze(0))
            loss2 = criterion(bluredB, imten2.reshape(D, H, W).unsqueeze(0).unsqueeze(0))
            loss = (loss1 + loss2) / 2

            optim.zero_grad()
            loss.backward()
            optim.step()

            mse_array[idx] = loss
            time_array[idx] = time.time()

            wandb.log({'mse': mse_array[idx]}, step=idx)

            scheduler.step()

        if not idx % config.verbose_freq or idx == config.niters - 1:
            # Process images and upload to wandb
            coords = utils.get_coords(D, H, W)
            im_estim = torch.zeros(coords.shape[0], 1, device='cuda')
            indices = torch.randperm(len(coords))
            with torch.no_grad():
                for b_idx in range(0, len(coords), maxpts):
                    b_indices = indices[b_idx:min(len(coords), b_idx + maxpts)]
                    b_coords = coords[b_indices, ...].to(config.device)
                    b_indices = b_indices.to(config.device)
                    pixelvalues = model(b_coords[None, ...]).squeeze()[:, None]

                    im_estim[b_indices, :] = pixelvalues

            im_log = im_estim.reshape(D, H, W).detach().cpu().numpy()
            im_log_normalized = normalize(im_log.astype(np.float32))

            # calculate PSNR and SSIM
            psnr_gt = skimage.metrics.peak_signal_noise_ratio(gt_normalized, im_log_normalized)
            ssim_gt = skimage.metrics.structural_similarity(gt_normalized, im_log_normalized)

            # log PSNR and SSIM to wandb
            run.log({"PSNR_gt": psnr_gt, "SSIM_gt": ssim_gt})

            con_recon = process_image(im_log)
            wandb.log({"Recon": wandb.Image(con_recon)})

            if idx == 0:
                con_A = process_image(im_ten.reshape(D, H, W).detach().cpu().numpy())
                con_B = process_image(im_ten2.reshape(D, H, W).detach().cpu().numpy())
                gt_log = process_image(
                    imProcessRGB_3D(gt, bit_depth=config.bit_depth).reshape(D, H, W).detach().cpu().numpy())

                wandb.log({
                    "ViewA": wandb.Image(con_A),
                    "ViewB": wandb.Image(con_B),
                    "GT": wandb.Image(gt_log)
                })

            if idx >= config.cycle_start:
                im_log_A = bluredA.squeeze(0).squeeze(0).detach().cpu().numpy()
                im_log_A_normalized = normalize(im_log_A.astype(np.float32))

                im_log_B = bluredB.squeeze(0).squeeze(0).detach().cpu().numpy()
                im_log_B_normalized = normalize(im_log_B.astype(np.float32))

                psnr_A = skimage.metrics.peak_signal_noise_ratio(im_normalized, im_log_A_normalized)
                ssim_A = skimage.metrics.structural_similarity(im_normalized, im_log_A_normalized)
                psnr_B = skimage.metrics.peak_signal_noise_ratio(im2_normalized, im_log_B_normalized)
                ssim_B = skimage.metrics.structural_similarity(im2_normalized, im_log_B_normalized)

                # log PSNR and SSIM to wandb
                run.log({"PSNR_A": psnr_A, "SSIM_A": ssim_A, "PSNR_B": psnr_B, "SSIM_B": ssim_B})

                deg_A = process_image(im_log_A)
                deg_B = process_image(im_log_B)

                wandb.log({
                    "degA": wandb.Image(deg_A),
                    "degB": wandb.Image(deg_B)
                })

        tbar.set_description('%.4e' % mse_array[idx])
        tbar.refresh()

    total_time = time.time() - tic
    nparams = utils.count_parameters(model)

    # Save model checkpoints
    torch.save(model.state_dict(), f'{run_id}_checkpoint_model_last.pth')
    model_artifact = wandb.Artifact(f'{run_id}_checkpoint_last', type='model')
    model_artifact.add_file(f'{run_id}_checkpoint_model_last.pth', name='model.pth')
    wandb.log_artifact(model_artifact)
    os.remove(f'{run_id}_checkpoint_model_last.pth')

    print('Total time %.2f minutes' % (total_time / 60))
    print('Total pararmeters: %.2f million' % (nparams / 1e6))

    run.finish()