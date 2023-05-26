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
nltk.download('words')
from nltk.corpus import words

def prepare_hyperparameters():
    return dict(
        project='relu',
        verbose=True,
        verbose_freq=400,
        niters=4000,
        lr=5e-3,
        expname='5_SPIMA_noAffine',
        expname2='5_SPIMB_noAffine',
        bit_depth=16,
        scale=1,
        omega0=30.0,
        sigma0=20.0,
        hidden_layers=2,
        hidden_features=256,
        in_features=3,
        out_features=1,
        maxpoints=int(2e5),
        device='cuda',
        nonlin='wire'
    )

def set_seeds(seed=7):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def get_random_word():
    return random.choice(words.words())

def init_wandb(hyperparameters):
    run_name = f"{hyperparameters['project']}_w{hyperparameters['omega0']}_s{hyperparameters['sigma0']}_{hyperparameters['lr']}_{get_random_word()}"
    run = wandb.init(config=hyperparameters, project=hyperparameters['project'], name=run_name)
    return wandb.config, run

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

def imProcessRGB_3D(im1, bit_depth=16):
    max_val = 2**bit_depth - 1
    imNorm = lambda x: (x - 0.5) * 2
    im1 = torch.FloatTensor(im1.astype(np.float32) * (1.0 / max_val)).cuda().reshape(D * H * W, 1)
    return imNorm(im1)

if __name__ == '__main__':

    set_seeds()
    hyperparameters = prepare_hyperparameters()
    config, run = init_wandb(hyperparameters)

    im = np.float32(tifffile.imread(f'data/{config.expname}.tiff'))
    im2 = np.float32(tifffile.imread(f'data/{config.expname2}.tiff'))

    print(f'[INFO] Image shape: {im.shape}')
    D, H, W = im.shape
    maxpts = min(D * H * W, config.maxpoints)

    # Get the mask for the desired slices
    slices = torch.arange(0, D, 1).cuda()  # 0, 10, 20, ..., 110
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
                    sidelength=max(D, H, W)).cuda()
    
    # Optimizer
    optim = torch.optim.Adam(lr=config.lr, params=model.parameters())
    
    # Schedule to 0.1 times the initial rate
    scheduler = LambdaLR(optim, lambda x: 0.2**min(x/config.niters, 1))

    criterion = torch.nn.MSELoss()
    
    # Create inputs
    coords = utils.get_coords(D, H, W)
    coords = coords.cuda()
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

        mse_array[idx] = train_loss/nchunks
        time_array[idx] = time.time()

        wandb.log({'mse': mse_array[idx], 'time': time_array[idx]}, step=idx)

        scheduler.step()

        if not idx % config.verbose_freq or idx == config.niters - 1:
            # Process images and upload to wandb
            coords = utils.get_coords(D, H, W)
            im_estim = torch.zeros(coords.shape[0], 1, device='cuda')
            indices = torch.randperm(len(coords))
            with torch.no_grad():
                for b_idx in range(0, len(coords), maxpts):
                    b_indices = indices[b_idx:min(len(coords), b_idx + maxpts)]
                    b_coords = coords[b_indices, ...].cuda()
                    b_indices = b_indices.cuda()
                    pixelvalues = model(b_coords[None, ...]).squeeze()[:, None]

                    im_estim[b_indices, :] = pixelvalues

            im_log = im_estim.reshape(D, H, W).detach().cpu().numpy()

            con_recon = process_image(im_log)
            con_A = process_image(im_ten.reshape(D, H, W).detach().cpu().numpy())
            con_B = process_image(im_ten2.reshape(D, H, W).detach().cpu().numpy())

            wandb.log({
                "Recon": wandb.Image(con_recon),
                "ViewA": wandb.Image(con_A),
                "ViewB": wandb.Image(con_B)
            })

        tbar.set_description('%.4e'%mse_array[idx])
        tbar.refresh()
        
    total_time = time.time() - tic
    nparams = utils.count_parameters(model)

    print('Total time %.2f minutes'%(total_time/60))
    print('Total pararmeters: %.2f million'%(nparams/1e6))

    run.finish()