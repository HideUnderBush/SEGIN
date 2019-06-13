import pdb
import os, os.path
import torch
import torchvision.models as tmodels
import numpy as np
import matplotlib.pyplot as plt
import Utils
#import VGG19
from models import FeatureExtractor

d = 16 
content_layers = 'relu5_1'

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.transpose(0,1)
    #x = x.view(x.size(0), 3, 256, 256)
    print('shape of x{}'.format(x.shape))
    return x

def norm_feat(feat):
    _, c, h, w = feat.shape
    for i in range(h):
        for j in range(w):
            #feat[:,0:c, i, j] = (feat[:, 0:c, i, j] - torch.min(feat[:, 0:c, i, j], dim=1)[0]) / (torch.max(feat[:, 0:c, i, j], dim=1)[0] - torch.min(feat[:, 0:c, i, j], dim=1)[0])
            feat[:,0:c, i, j] = (feat[:, 0:c, i, j] - torch.min(feat[:, 0:c, i, j], dim=1)[0]) / (torch.max(feat[:, 0:c, i, j], dim=1)[0] - torch.min(feat[:, 0:c, i, j], dim=1)[0])
    return feat

def reconstruct(img, max_sim):
    scale = 256 // d 
    img = torch.squeeze(img,0)
    res = torch.zeros(img.shape)
    print(res.shape)
    for i in range(d):
        for j in range(d):
            x = int(max_sim[i][j] % d) 
            y = int(max_sim[i][j] / d) 
            res[0:3, scale * i:scale * (i + 1), scale * j:scale * (j + 1)] = img[0:3, scale * y:scale * (y + 1), scale * x:scale * (x + 1)]
    return torch.transpose(res, 0, 2)

torch.cuda.set_device(1)
pathA = '/home/chloe/Pytorch_textureGAN_Changed/training_shoes_pretrain_total/val_img/wendy/'
pathB = '/home/chloe/Pytorch_textureGAN_Changed/training_shoes_pretrain_total/val_skg/wendy/'
pathOut = '/home/chloe/Pytorch_textureGAN_Changed/training_shoes_pretrain_total/val_war_4/wendy/'

layers_map = {'relu5_1':'29','relu4_2': '22', 'relu4_1':'20', 'relu2_1': '6', 'relu2_2': '8', 'relu3_1':'11', 'relu3_2': '13','relu1_2': '3'}
feat_model = tmodels.vgg19(pretrained=True).cuda()
#feat_model = VGG19()
extract_content = FeatureExtractor(feat_model.features, [layers_map[x.strip()] for x in content_layers.split(',')])


item_num = len([name for name in os.listdir(pathA) if os.path.isfile(os.path.join(pathA, name))])
#for i in range(1, 2):
for i in range(1, item_num):
#for name in os.listdir(pathA):
    
    ran_match = np.random.randint(item_num)+1
    img_path = pathA + str(i) + '_AB.jpg'
    #skg_path = pathB + str(i) + '_AB.jpg'
    skg_path = pathB + str(ran_match) + '_AB.jpg'
    #img_path = '/home/chloe/Desktop/content.jpg'
    #skg_path = '/home/chloe/Desktop/style.jpg'

    img = Utils.load_image(img_path=img_path, to_array=True, to_variable=True).cuda()
    skg = Utils.load_image(img_path=skg_path, to_array=True, to_variable=True).cuda()

    img_feat = extract_content(img)[0]
    skg_feat = extract_content(skg)[0]
    print('img feat shape: ', img_feat.shape)
    print('skg feat shape: ', skg_feat.shape)

    img_feat = img_feat.view(512, d*d)
    img_feat = img_feat / torch.norm(img_feat, dim=0, keepdim=True)
    skg_feat = skg_feat.view(512, d*d)
    skg_feat = skg_feat / torch.norm(skg_feat, dim=0, keepdim=True)
    print('img feat shape: ', img_feat.shape)
    #print(img_feat[:,0])
    print('skg feat shape: ', skg_feat.shape)

    sim_map = torch.matmul(torch.transpose(img_feat, 0, 1), skg_feat).view(d*d, d, d)
    print('sim map shape: ', sim_map.shape)

    max_sim = torch.argmax(sim_map, 0, keepdim=False)
    print('max sim shape: ',max_sim.shape)

    war = reconstruct(img, max_sim)
    war = to_img(war)
    war_path = pathOut + str(i) + '_AB.jpg'
    #war_path = '/home/chloe/Desktop/war.jpg'
    plt.imsave(war_path, war)

