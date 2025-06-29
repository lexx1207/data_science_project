import warnings
warnings.simplefilter("ignore", UserWarning)
import splitfolders
import math
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random  
from pathlib import Path
from PIL import Image
from tqdm.notebook import tqdm
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import models.adainnet
from models.adainnet import Model, VGGEncoder, RC, Decoder, denorm
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torchvision import models

def adain(CONTENT_IMAGE ='example_image/content/000000000298.jpg', STYLE_IMAGE = 'example_image/style/7.jpg',OUT_DIR = 'out_content'):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    PATH_TO_MODEL = 'chkpnts/full_model.pth'
    model = torch.load(PATH_TO_MODEL, weights_only=False)
    model = model.to(device)
    model.eval()
    content_image = Image.open(CONTENT_IMAGE).convert("RGB") 
    style_image = Image.open(STYLE_IMAGE).convert("RGB") 

    transform_im = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
    
    content_image = transform_im(content_image).unsqueeze(0).to(device)
    style_image = transform_im(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.generate(content_image, style_image)
        out = denorm(out, device).detach().cpu()
        ndarr = out.squeeze(0).mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        out_img = Image.fromarray(ndarr)
        file_name = "output.png"
        path = os.path.join(OUT_DIR, file_name)
        out_img.save(path, quality=100)
        
