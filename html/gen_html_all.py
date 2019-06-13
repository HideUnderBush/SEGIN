import os
import numpy as np

def get_ref_gt_names(name):
    num = name.split('_')[0]
    gt_name = path_gt + str(num) + '_AB.jpg'
    ref_name = path_ref + str(num) + '_AB.jpg'
    return ref_name, gt_name

path_ref = '/home/chloe/Pytorch_textureGAN_Changed/training_shoes_pretrain_total/val_img/wendy/'
#path_res = '../Test_Results/test_results_self_snresb_style/'
path_res = '../test_results_caseall_mask/'
path_gt = '/home/chloe/Pytorch_textureGAN_Changed/training_shoes_pretrain_total/val_img/wendy/'

res_html = './res_gan_caseall_mask.html'
if os.path.exists(res_html):
    os.remove(res_html)
    os.mknod(res_html)

#f_match = open('./match.txt','r') 
#match = f_match.readlines()
#print(len(match))

f = open(res_html,'a')
f.write('!DOCTYPE html\n <html>\n <body>\n')

f.write('<h2>{}</h2>'.format(path_res.split('/')[1]))

for k in range(1, 50):
    print(k)
    for i in range(1,200):
        #print(i)
        ref = path_ref + str(k) + '_AB.jpg' 
        res = path_res + 'image_' + str(k) + '_' +  str(i)  + '.jpg'  
        gt = path_gt + str(i) + '_AB.jpg' 
        f.write('<p>\n')
        f.write('ref {} img {}\n'.format(k, i))
        f.write('</p>\n')

        f.write('<p>\n')
        f.write('<img src=\"{}\" alt=\"Trulli\" width=\"300\" height=\"333\">\n'.format(ref))
        f.write('<img src=\"{}\" alt=\"Trulli\" width=\"900\" height=\"333\">\n'.format(res))
        f.write('<img src=\"{}\" alt=\"Trulli\" width=\"300\" height=\"333\">\n'.format(gt))
        f.write('</p>\n')

f.write('</body>\n </html>\n')
f.close()
