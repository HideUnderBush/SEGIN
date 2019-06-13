import pdb
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from utils import transforms as custom_transorms
from torch.utils.data import SequentialSampler as SequentialSampler 
from dataloader.imfol import ImageFolder
from models import network, network_nonlocal, network_nonlocal_nfc, network_nonlocal_test
import numpy as np
import os

def gen_input_mask(war, skg):
    w, h = war.size()[1:3]
    #print(img.shape)
    input_texture = war.clone() # torch.ones(img.size())*(1)
    input_sketch = skg.clone()# L channel from skg
    input_mask = torch.ones(skg[0:1,:,:].size())*(-1)
    #input_texture = torch.ones(img.size())*(-1)

    #input_mask[:,:,:] = 1 # no mask at all, only need this struture for test
    #input_mask[img[0,:,:].unsqueeze(0)<0.9] = 1
    input_mask[war[0:1,:,:]<0.93] = 1
    input_texture = input_texture * input_mask.cuda()

    #input_latent = torch.randn(skg[0:1,:,:].size())
    #input_latent[war[0:1,:,:]<0.93] = -1

    #return torch.cat((input_sketch.cpu().float(), input_texture.cpu().float(), input_latent), 0)

    return torch.cat((input_sketch.cpu().float(), input_texture.cpu().float()), 0)


def stack_input(war, skg):
    bs, c, w, h = war.shape
    res = torch.ones([bs, 4, w, h])
    #res = torch.ones([bs, 5, w, h])
    for i in range(bs):
        res[i,:,:,:] = gen_input_mask(war[i,:,:,:], skg[i,:,:,:])
    return res

def stack_input_seg(war, skg, seg):
    bs, c, w, h = war.shape
    #res = torch.ones([bs, 7, w, h])
    res = torch.cat((skg, war, seg[:,0:1,:,:]),1)
    return res

def stack_input_skg(war, skg):
    bs, c, w, h = war.shape
    #res = torch.ones([bs, 7, w, h])
    res = torch.cat((skg, war),1)
    return res

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    #x = x.view(x.size(0), 3, 256, 256)
    x = x.view(3, 3, 256, 256)
    print('shape of x{}'.format(x.shape))
    return x


def to_img_single(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    #x = x.view(x.size(0), 3, 256, 256)
    x = x.view(3, 3, 256, 256)
    x_single = x[2:3, :,:,:]
    print('shape of x{}'.format(x.shape))
    return x_single 

def first_three(x):
    return(x[0:2])

# Prepare a folder to store the test results (img)
#if not os.path.isdir('./test_results_self_snres6_style_lessfeat'):
#    os.mkdir('./test_results_self_snres6_style_lessfeat')
torch.cuda.set_device(1)
if not os.path.isdir('./test_results_case1_simabl'):
    os.mkdir('./test_results_case1_simabl')

batch_size = 1 

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5,0.5))
])

#war_path = '/home/chloe/Pytorch_textureGAN_Changed/training_shoes_pretrain_total/val_war_3/wendy/'
war_path = '/home/chloe/Pytorch_textureGAN_Changed/training_shoes_pretrain_total/val_img/wendy/'
#war_path = '/home/chloe/all_warp_coarse/warp164/'
war_list = os.listdir(war_path)
sorted(war_list)
img_path = '/home/chloe/Pytorch_textureGAN_Changed/training_shoes_pretrain_total/val_img/wendy/'
img_list = os.listdir(img_path)
sorted(img_list)
skg_path = '/home/chloe/Pytorch_textureGAN_Changed/training_shoes_pretrain_total/val_skg/wendy/'
skg_list = os.listdir(skg_path)
sorted(skg_list)
seg_path = '/home/chloe/Pytorch_textureGAN_Changed/training_shoes_pretrain_total/val_seg/wendy/'
seg_list = os.listdir(seg_path)
sorted(seg_list)

#test_dataset = ImageFolder('val','/home/chloe/Pytorch_textureGAN_Changed/training_shoes_pretrain_total', img_transform)
#test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=SequentialSampler(test_dataset))

#model_G = network_nonlocal_test.generator().cuda()
model_G = network_nonlocal_nfc.generator().cuda()
#model_G = network_nonlocal.generator().cuda()
model_G.load_state_dict(torch.load('./generated_models_I2IVratial/conv_generator_19_8000.pth')) # 14_8000.pth works well
#model_G.load_state_dict(torch.load('./Generated_Models/new_sketch_new_warp/conv_generator_4_8000.pth')) # 14_8000.pth works well
###model_G.load_state_dict(torch.load('./remote/conv_generator_11_8000.pth')) # 14_8000.pth works well
#model_G.load_state_dict(torch.load('./conv_generator_rand.pth'))
#model_G.load_state_dict(torch.load('/home/chloe/autoencoderTest/generated_models_harderMorefeat/conv_generator_8_0.pth'))

# network
#for i, data in enumerate(test_loader):
for i in range(len(war_list)):
    img_f = open(img_path + img_list[i], 'rb') 
    img = Image.open(img_f).convert('RGB')
    img = img_transform(img)
    img = img.unsqueeze(0)

    #war_f = open(war_path + str(i+1) + '_AB.jpg', 'rb') 
    war_f = open(war_path + str(93) + '_AB.jpg', 'rb') 
    war = Image.open(war_f).convert('RGB')
    war = img_transform(war)
    war = war.unsqueeze(0)
    skg_f = open(skg_path + str(i+1) + '_AB.jpg', 'rb') 
    skg = Image.open(skg_f).convert('RGB')
    skg = img_transform(skg)
    skg = skg.unsqueeze(0)

    seg_f = open(seg_path + str(i+1) + '_AB.jpg', 'rb') 
    seg = Image.open(seg_f).convert('RGB')
    seg = img_transform(seg)
    seg = seg.unsqueeze(0)

    #img, skg, war = data
    img = Variable(img).cuda()
    skg = Variable(skg).cuda()
    war = Variable(war).cuda()
    seg = Variable(seg).cuda()
    input_ = Variable(stack_input(war, skg[:,0:1,:,:])).cuda()
    #input_ = stack_input_seg(war, skg, seg).cuda()
    #input_ = stack_input_skg(war, skg[:,0:1,:,:]).cuda()
    output, out_seg= model_G(input_)
    #output = output * out_seg
    output_= torch.cat((skg.cpu(), war.cpu(), output.cpu()),1)
    ####pic = to_img(output_.cpu().data) 
    pic = to_img_single(output_.cpu().data) 
    #save_image(pic, './test_results_self_snres6_style_lessfeat/image_{}.jpg'.format(i+1))
    #save_image(pic, './test_results_case1/image_{}.jpg'.format(i+1))
    save_image(pic, './test_results_case1_simabl/image_{}.jpg'.format(i+1))
