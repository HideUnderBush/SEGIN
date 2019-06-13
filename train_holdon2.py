import pdb
import numpy as np
import torch
import torchvision
import torchvision.models as tmodels
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from utils import transforms as custom_transforms
#from torchvision.datasets import MNIST
from dataloader.imfol import ImageFolder
from torch.optim.lr_scheduler import StepLR
from models import network, network_nonlocal, network_nonlocal_nfc, network_nonlocal_test, srnetwork, SNnetwork, FeatureExtractor
from tensorboardX import SummaryWriter
import os
import random;
from functools import reduce

if not os.path.exists('./I2IVratial_img'):
    os.mkdir('./I2IVratial_img')
if not os.path.exists('./generated_models_I2IVratial'):
    os.mkdir('./generated_models_I2IVratial')

def randomPicker(howMany, *ranges):
    mergedRange = reduce(lambda a, b: a + b, ranges);
    ans = [];
    for i in range(howMany):
        ans.append(random.choice(mergedRange));
    return ans;

    #x = randomPicker(1, list(range(-10, -5)), list(range(0,2))) # Usage

def gen_input_mask(war, skg):
    w, h = war.size()[1:3]
    input_texture = war.clone()  # torch.ones(img.size())*(1)
    input_sketch = skg.clone()# L channel from skg
    input_mask = torch.ones(skg[0:1,:,:].size())*(-1)
    #input_texture = torch.ones(img.size())*(-1)

    #input_mask[:,:,:] = 1 # no mask at all, only need this struture for test
    #input_mask[img[0,:,:].unsqueeze(0)<0.9] = 1
    input_mask[war[0:1,:,:]<0.93] = 1
    input_texture = input_texture * input_mask.cuda()

    #return torch.cat((input_sketch.cpu().float(), input_texture.cpu().float(), input_mask), 0)
    return torch.cat((input_sketch.cpu().float(), input_texture.cpu().float()), 0)
 
def stack_input(war, skg):
    bs, c, w, h = war.shape
    res = torch.ones([bs, 4, w, h])
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

def random_white_space(img, batch_size, ini_shift, type_):
    # center of shoes is around 75
    if type_ == 'white':
        img = preprocess_white(img)
    elif type_ == 'shift':
        img = preprocess(img, ini_shift)
    #psize = 20 
    psize = np.random.randint(15, 20) 
    for k in range(batch_size):
        #for i in range(10,30):
        #for i in range(10,20):
        if type_ == 'white':
            a = 10 
            b = 20
        elif type_ == 'shift':
            a = 1
            b = 18 
        for i in range(a,b):
            if type_ == 'white':
                randcenx = np.random.randint(35, 220, 2)
            elif type_ == 'shift':
                randcenx = np.random.randint(60, 195, 2)
            #randceny = np.random.randint(30, 230) # this is the original setting
            x0 = randcenx[0] - psize
            x1 = randcenx[0] + psize
            y0 = randcenx[1] - psize
            y1 = randcenx[1] + psize
            img[k, :, x0:x1, y0:y1] = 1
    return img

def local_crop(img1, img2, epoch, iteration):
    #TODO dynamic local crop based on steps
    #if (epoch > 0 or iteration > 1000):
    #    c_size = 10 
    #    xc = np.random.randint(40, 230) 
    #    yc = np.random.randint(40, 230) 
    #else:
    c_size = 50 
    xc = np.random.randint(70,200,2) 
    #yc = np.random.randint(70,200) 
    bs, c, w, h = img.shape
    x0 = xc[0] - c_size
    x1 = xc[0] + c_size 
    y0 = xc[1] - c_size 
    y1 = xc[1] + c_size 
    res1 = img1[:,:, x0:x1, y0:y1].clone()
    res2 = img2[:,:, x0:x1, y0:y1].clone()
    return res1, res2

def preprocess(war, ini):
    # add this inside random_white_space function
    # select a patch
    res = war.clone()
    #for i in range(0, 30): # repeat this 20 times this is the first setting
    for i in range(5, 30): # repeat this 20 times
        psize = np.random.randint(30, 35) # prepare 20 x 20 small patches
        xc = np.random.randint(50, 200, 2)
        x_ori0 = int(xc[0] - psize) 
        x_ori1 = int(xc[0] + psize)
        y_ori0 = int(xc[1] - psize)
        y_ori1 = int(xc[1] + psize)
        #shift_times = np.random.randint(10)
        shift_times = np.random.randint(ini, 20)
        for j in range(shift_times):
            #shift_range = np.random.randint(-15, 15, 2) 
            shift_range = randomPicker(2, list(range(6, 10)), list(range(-10, -6))) 
            x_new = xc[0] + shift_range[0]    
            y_new = xc[1] + shift_range[1]    
            x0 = x_new - psize
            x1 = x_new + psize
            y0 = y_new - psize
            y1 = y_new + psize
            if j % 5 == 0:
                res[:,:,y0:y1,x0:x1] = war[:,:,x_ori0:x_ori1,y_ori0:y_ori1] # 45 degree flip
            else:
                res[:,:,x0:x1,y0:y1] = war[:,:,x_ori0:x_ori1,y_ori0:y_ori1] 

    return res


def preprocess_white(war):
    # add this inside random_white_space function
    # select a patch
    res = war.clone()
    for i in range(0, 20): # repeat this 20 times
        psize = np.random.randint(30, 35) # prepare 20 x 20 small patches
        xc = np.random.randint(50, 210, 2)
        x_ori0 = int(xc[0] - psize) 
        x_ori1 = int(xc[0] + psize)
        y_ori0 = int(xc[1] - psize)
        y_ori1 = int(xc[1] + psize)
        shift_times = np.random.randint(10)
        for j in range(shift_times):
            shift_range = np.random.randint(-10, 10, 2) 
            x_new = xc[0] + shift_range[0]    
            y_new = xc[1] + shift_range[1]    
            x0 = x_new - psize
            x1 = x_new + psize
            y0 = y_new - psize
            y1 = y_new + psize
            res[:,:,x0:x1,y0:y1] = war[:,:,x_ori0:x_ori1,y_ori0:y_ori1] 
    return res

def to_img_summary(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    #x = x.view(x.size(0), 3, 50, 50)
    #print('shape of x{}'.format(x.shape))
    return x



def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)
    print('shape of x{}'.format(x.shape))
    return x

def GramMatrix(input_):
    b, c, h, w = input_.shape 
    F = input_.view(b, c, h*w)
    G = torch.bmm(F, F.transpose(1, 2))
    G.div_(h * w)
    return G

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

# parameter (move is to argparser in the future)
torch.cuda.set_device(1)
network_type = 'sn'
input_type = 'no_seg'
num_epochs = 20 
batch_size = 5 
learning_rate = 2e-4 # default is 2e-4, default for discriminator is 1e-5
beta1= 0.9
beta2= 0.999
content_layers = 'relu3_2'
#style_layers = 'relu2_1, relu3_1, relu4_1'
style_layers = 'relu2_1'
#layers_map = {'relu4_2': '22', 'relu4_1':'21', 'relu2_1': '7', 'relu3_1':'12', 'relu3_2': '13','relu1_2': '4'}
layers_map = {'relu4_2': '22', 'relu4_1':'20', 'relu2_1': '6', 'relu2_2': '8', 'relu3_1':'11', 'relu3_2': '13','relu1_2': '3'}

writer = SummaryWriter()

img_transform = transforms.Compose([
    #custom_transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
'''
transforms_list = [
    custom_transforms.RandomSizedCrop(image_size, resize_min, resize_max),
    custom_transforms.toTensor()
]
transforms_ = custom_transforms.Compose(transforms_list)
'''


train_dataset = ImageFolder('train', '../Pytorch_textureGAN_Changed/training_shoes_pretrain_total/', img_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_loader_gan = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#print('SAHPE {}'.format(train_loader_gan.__iter__().__next__()[0].shape))
#train = train_loader.__iter__().__next__()[0]
#img_size=train.size()[2]

# network
if network_type == 'srgan':
    model_G = srnetwork.Generator(16, 1).cuda()
    model_D = srnetwork.Discriminator().cuda()
    model_local_D = srnetwork.Discriminator().cuda()
elif network_type == 'sn':
    if input_type == 'no_seg':
        #model_G = network_nonlocal.generator().cuda()
        model_G = network_nonlocal_nfc.generator().cuda()
        #model_D = network_nonlocal.discriminator_snIns().cuda()
        model_D = network_nonlocal_test.discriminator_snIns().cuda()
        #model_local_D = network_nonlocal.discriminator_snIns().cuda()
    else:
        #model_G = network_allSN.generator().cuda()
        model_G = network.generator().cuda()
        #model_D = SNnetwork.Discriminator(3, 64).cuda()
        model_D = network.discriminator_snIns().cuda()
        #model_local_D = SNnetwork.Discriminator(3, 64).cuda()
        model_local_D = network.discriminator_snIns().cuda()
elif network_type == 'nlayerD':
    model_G = network.generator().cuda()
    model_D = network.NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).cuda()
    model_local_D = network.NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).cuda()
else: 
    model_G = network.generator().cuda()
    model_D = network.discriminator().cuda()
    model_local_D = network.discriminator().cuda()
feat_model = tmodels.vgg19(pretrained=True).cuda()
extract_content = FeatureExtractor(feat_model.features, [layers_map[x.strip()] for x in content_layers.split(',')])
extract_style = FeatureExtractor(feat_model.features, [layers_map[x.strip()] for x in style_layers.split(',')])

# loss criterion
BCE_loss = nn.BCELoss().cuda()
MSE_loss = nn.MSELoss().cuda()
TV_loss = TVLoss().cuda()
criterion = nn.L1Loss().cuda()

# Adam optimizer
#G_optimizer = torch.optim.Adam(model_G.parameters(), lr=learning_rate, weight_decay=1e-5)
G_optimizer = torch.optim.Adam(model_G.parameters(), lr=learning_rate, betas=(beta1, beta2))
#G_local_scheduler = StepLR(G_optimizer, step_size=10, gamma=0.8)
#D_optimizer = torch.optim.Adam(model_D.parameters(), lr=learning_rate*0.5, weight_decay=1e-5)
D_optimizer = torch.optim.Adam(model_D.parameters(), lr=1.3e-5, betas=(beta1, beta2))
#D_scheduler = StepLR(D_optimizer, step_size =10, gamma=0.8)
#D_local_optimizer = torch.optim.Adam(model_local_D.parameters(), lr=learning_rate*0.5, weight_decay=1e-5)
#D_local_optimizer = torch.optim.Adam(model_local_D.parameters(), lr=2e-5, betas=(beta1, beta2))
#D_local_scheduler = StepLR(D_local_optimizer, step_size =10, gamma=0.8)

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):

        img, skg, war, seg = data
        #img, skg, _, seg = data # we don't need real war imgage in training stage
        img_real, skg_real, _, _ = train_loader_gan.__iter__().__next__()
        img = Variable(img).cuda()
        img_real = Variable(img_real).cuda()
        if input_type == 'no_seg':
            skg = Variable(skg[:,0:1,:,:]).cuda()
            skg_real = Variable(skg_real[:,0:1,:,:]).cuda()
        else:
            skg = Variable(skg).cuda()
        #if np.random.random_sample() > 0.5: 
        #    war = Variable(war).cuda()
        #else: 
        #    war = Variable(torch.ones(war.size())).cuda()

        # randomly add more white space in the training stage
        #if i % 2 == 0: #this is the first setting
        if i % 4 == 0:
            war = random_white_space(war, batch_size, 0, 'white') 
        else:
            war = random_white_space(img.clone(), batch_size, 5, 'shift') 
        war = Variable(war).cuda()
        if input_type == 'no_seg':
            seg = Variable(seg[:,0:1,:,:]).cuda()
        else:
            seg = Variable(seg).cuda()
        #noise = Variable(img.data.new(img.size()).normal_(0,0.05)) 

        #input_ = torch.autograd.Variable(stack_input(war, skg), requires_grad=True).cuda()
        if input_type == 'no_seg':
            input_ = Variable(stack_input(war, skg)).cuda()
            #input_ = Variable(stack_input_skg(war, skg)).cuda()
        else:
            input_ = Variable(stack_input_seg(war, skg, seg)).cuda()
        output, out_seg = model_G(input_)
        #print('shape of output{}'.format(output.shape))

        # D loss
        model_D.zero_grad()
        #x_ = img + noise
        #x_ = img 
        #img_x = Variable(x_.cuda()) #no flip or crop here, TODO
        D_real_result = model_D(img_real).squeeze()
        #D_real_loss = MSE_loss(D_real_result, Variable(torch.ones(D_real_result.size()).cuda()))
        if epoch % 2 == 0:
            D_real_loss = BCE_loss(D_real_result, Variable(torch.zeros(D_real_result.size()).cuda()))
        else:
            D_real_loss = BCE_loss(D_real_result, Variable(torch.ones(D_real_result.size()).cuda()))

        D_fake_result = model_D(output.detach()).squeeze()
        #D_fake_loss = MSE_loss(D_fake_result, Variable(torch.zeros(D_fake_result.size()).cuda()))
        if epoch % 2 == 0: 
            D_fake_loss = BCE_loss(D_fake_result, Variable(torch.ones(D_fake_result.size()).cuda()))
        else:
            D_fake_loss = BCE_loss(D_fake_result, Variable(torch.zeros(D_fake_result.size()).cuda()))

        #D_train_loss = (D_real_loss + D_fake_loss) * 0.5
        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward(retain_graph=True)
        D_optimizer.step()


        # local D loss
        ##model_local_D.zero_grad()

        ###img_x_crop, output_crop = local_crop(img_real, output, epoch, i)
        ###D_local_real_result = model_local_D(img_x_crop).squeeze()
        #D_local_real_loss = MSE_loss(D_local_real_result, Variable(torch.ones(D_local_real_result.size()).cuda()))
        ###D_local_real_loss = BCE_loss(D_local_real_result, Variable(torch.ones(D_local_real_result.size()).cuda()))

        ####D_local_fake_result = model_local_D(output_crop.detach()).squeeze()
        ####D_local_fake_loss = BCE_loss(D_local_fake_result, Variable(torch.zeros(D_local_fake_result.size()).cuda()))

        #D_local_train_loss = (D_local_real_loss + D_local_fake_loss) * 0.5
        ###D_local_train_loss = D_local_real_loss + D_local_fake_loss
        ##D_local_train_loss.backward(retain_graph=True)
        ##D_local_optimizer.step()


        # feature loss
        out_feat = extract_content(output)[0]
        img_feat = extract_content(img)[0]
        feat_loss = MSE_loss(out_feat, img_feat) * 5e-6  # this is for relu3_2 or relu_4_2

        # style loss
        #out_style_feat = extract_style(output)[0]
        #img_style_feat = extract_style(img)[0]
        #out_style = GramMatrix(out_style_feat)
        #img_style = GramMatrix(img_style_feat)
        #style_loss = MSE_loss(out_style, img_style) * 0.03     # default is 0.01 MSE should be the suitable ones.

        #local L1 loss
        #local_l1_loss = criterion(output_crop, img_x_crop) * 100 

        # accurate seg loss
        acc_seg_loss = criterion(out_seg * output, seg * img) * 10 


        # G loss and total loss
        model_G.zero_grad()
        D_result = model_D(output).squeeze()
        ###D_local_result = model_local_D(output_crop).squeeze()
        #G_train_loss = MSE_loss(D_result, Variable(torch.ones(D_result.size()).cuda())) 

        # flip the flag sometimes
        if epoch % 2 == 0:
            G_train_loss = BCE_loss(D_result, Variable(torch.zeros(D_result.size()).cuda())) 
        else:
            G_train_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda())) 
        #G_local_train_loss = MSE_loss(D_local_result, Variable(torch.ones(D_local_result.size()).cuda())) 
        ####G_local_train_loss = BCE_loss(D_local_result, Variable(torch.ones(D_local_result.size()).cuda())) 
        l1_loss = criterion(output, img) * 100
        seg_loss = criterion(out_seg, seg) * 100
        tv_loss = TV_loss(output) * 10 
        #G_total_loss = G_train_loss + G_local_train_loss + feat_loss + l1_loss + local_l1_loss + style_loss
        G_total_loss = G_train_loss + l1_loss + seg_loss + feat_loss + tv_loss + acc_seg_loss
        G_total_loss.backward()
        G_optimizer.step()



    # ===================log========================
        print('epoch [{}/{}], iteration:{}, total_loss:{:.4f}, D_fake_res{:.4f}, D_real_res{:.4f}'.format(epoch+1, num_epochs, i, G_total_loss.data[0], D_fake_result.data[0], D_real_result.data[0]))


        if i % 5 == 0:
            #print('G_train_loss:{:.4f}, G_local_train_loss:{:.4f}, l1_loss:{:.4f}'.format(G_train_loss.data[0], G_local_train_loss.data[0], 100*l1_loss.data[0]))
            iter_total =int(49825 / batch_size)
            writer.add_scalar('data/err11_total',G_total_loss.item(), epoch*iter_total+i)
            writer.add_scalar('data/err1_G',G_train_loss.item(),epoch*iter_total+i)
            #writer.add_scalar('data/err3_local_G',G_local_train_loss.item(), epoch*iter_total+i)
            #writer.add_scalar('data/err4_local_D',D_local_train_loss.item(),epoch*iter_total+i)
            writer.add_scalar('data/err2_D',D_train_loss.item(), epoch*iter_total+i)
            writer.add_scalar('data/err5_l1',l1_loss.item(), epoch*iter_total+i)
            writer.add_scalar('data/err8_seg',seg_loss.item(), epoch*iter_total+i)
            writer.add_scalar('data/err10_acc_seg', acc_seg_loss.item(), epoch*iter_total+i)
            #writer.add_scalar('data/err_local_l1',local_l1_loss.item(), epoch*iter_total+i)
            writer.add_scalar('data/err6_feat',feat_loss.item(), epoch*iter_total+i)
            #writer.add_scalar('data/err7_style',style_loss.item(), epoch*iter_total+i)
            writer.add_scalar('data/err9_tv',tv_loss.item(), epoch*iter_total+i)
            #writer.add_scalar('data/err10_aseg',acc_seg_loss.item(), epoch*iter_total+i)
            #writer.add_image('image/0out_style_feat', to_img_summary(out_style_feat[0, 0:3, :,:]), epoch*iter_total+i)
            #writer.add_image('image/0img_style_feat', to_img_summary(img_style_feat[0, 0:3, :,:]), epoch*iter_total+i)
            #writer.add_image('image/1out_feat', to_img_summary(out_feat[0, 0:3, :,:]), epoch*iter_total+i)
            #writer.add_image('image/1img_feat', to_img_summary(img_feat[0, 0:3, :,:]), epoch*iter_total+i)
            writer.add_image('image/3img', to_img_summary(img[0]), epoch*iter_total+i)
            writer.add_image('image/5war', to_img_summary(war[0]), epoch*iter_total+i)
            writer.add_image('image/5skg', to_img_summary(skg[0]), epoch*iter_total+i)
            #writer.add_image('image/4seg', to_img_summary(seg[0]), epoch*iter_total+i)
            writer.add_image('image/3out', to_img_summary(output[0]), epoch*iter_total+i)
            #writer.add_image('image/4out_seg', to_img_summary(out_seg[0]), epoch*iter_total+i)
            #writer.add_image('image/2crop_img', to_img_summary(img_x_crop[0]), epoch*iter_total+i)
            #writer.add_image('image/2crop_out', to_img_summary(output_crop[0]), epoch*iter_total+i)

        if i % 200 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './I2IVratial_img/image_{}_{}.jpg'.format(epoch,i))
        if i % 4000 == 0:
            torch.save(model_G.state_dict(), './generated_models_I2IVratial/conv_generator_{}_{}.pth'.format(epoch, i))

    #G_local_scheduler.step()
    #D_local_scheduler.step()
    #D_scheduler.step()
