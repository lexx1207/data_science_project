import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as utils
from utils.utils import img_resize, load_segment
import numpy as np
from models.RevResNet import RevResNet
def capvstn(CONTENT_IMAGE = '/mount/src/data_science_project/CV_project_4/example_image/content/000000000298.jpg', STYLE_IMAGE = '/mount/src/data_science_project/CV_project_4/example_image/style/7.jpg',OUT_DIR = '/mount/src/data_science_project/CV_project_4/out_content'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    out_dir = OUT_DIR
    MAX_SIZE = 1280



    # Reversible Network
    
    RevNetwork = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4, hidden_dim=64, sp_steps=1)
    PATH_TO_MODEL = '/mount/src/data_science_project/CV_project_4/chkpnts/model_image.pt'
    state_dict = torch.load(PATH_TO_MODEL)
    RevNetwork.load_state_dict(state_dict['state_dict'])
    RevNetwork = RevNetwork.to(device)
    RevNetwork.eval()
    
    
    # Transfer module
    from models.cWCT import cWCT
    cwct = cWCT()
    
    
    content = Image.open(CONTENT_IMAGE).convert('RGB')
    style = Image.open(STYLE_IMAGE ).convert('RGB')
    
    ori_csize = content.size
    
    content = img_resize(content, MAX_SIZE, down_scale=RevNetwork.down_scale)
    style = img_resize(style, MAX_SIZE, down_scale=RevNetwork.down_scale)
    
    
    content_seg = None     # default
    style_seg = None     # default
    
    
    content = transforms.ToTensor()(content).unsqueeze(0).to(device)
    style = transforms.ToTensor()(style).unsqueeze(0).to(device)
    
    
    # Stylization
    with torch.no_grad():
        # Forward inference
        z_c = RevNetwork(content, forward=True)
        z_s = RevNetwork(style, forward=True)
    
       
        z_cs = cwct.transfer(z_c, z_s, content_seg, style_seg)
    
        # Backward inference
        stylized = RevNetwork(z_cs, forward=False)
    
    
    # save stylized
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cn = os.path.basename(CONTENT_IMAGE)
    sn = os.path.basename(STYLE_IMAGE)
    #file_name = "%s_%s.png" % (cn.split(".")[0], sn.split(".")[0])
    file_name = "output.png"
    path = os.path.join(out_dir, file_name)
    
    # stylized = transforms.Resize((ori_csize[1], ori_csize[0]), interpolation=Image.BICUBIC)(stylized)    # Resize to original size
    grid = utils.make_grid(stylized.data, nrow=1, padding=0)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    out_img = Image.fromarray(ndarr)
    
    out_img.save(path, quality=100)
    