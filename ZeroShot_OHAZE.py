import torch
import torchvision
import imageio
import model
import argparse
import os
from psnr import ssim
from cv2.ximgproc import guidedFilter
from utils.utils import _np2Tensor, _augment, psnr, t_matting, write_log
import torch.nn as nn
import torchvision.transforms as trans
from tqdm import tqdm
import math
import torchvision.transforms.functional as TF
from torchvision import transforms
from model.highmerge import Lap_Pyramid_Conv,padding_image,resize_and_pad_image
import numpy as np
from PIL import Image
from model.Loss import SSIMLoss
from model.CR import ContrastLoss
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


def Transf(image, H,W):
    transfomer = trans.Compose([
        trans.Resize((H, W))
    ])
    img = transfomer(image)
    return img

def TransF(image,H,W):
    transfomer = trans.Compose([
        trans.Resize((H, W))
    ])
    img = transfomer(image)
    return img

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1: #0.02
        m.weight.data.normal_(0.0, 0.001)
    if classname.find('Linear') != -1: #0.02
        m.weight.data.normal_(0.0, 0.001)

parser = argparse.ArgumentParser(description='Single Image Dehazing')
parser.add_argument('--TestFolderPath', type=str, default='./datasets/O-HAZE', help='Hazy Image folder name') 
parser.add_argument('--SavePath', type=str, default='./results/OHAZE-HFCM', help='SavePath Name')
args = parser.parse_args()

itr_no = 10000
l1_loss = nn.MSELoss()

criterionSsim = SSIMLoss()
criterion = torch.nn.MSELoss()
criterionP = torch.nn.L1Loss()
criterionC = ContrastLoss(device="cuda:2", ablation=True)
scaler = GradScaler()
def test(args):
    input_img = os.listdir(args.TestFolderPath+'/hazy/')
    input_img.sort()
    os.makedirs(args.SavePath+'/', exist_ok=True)
    
    total_psnr, total_ssim = 0, 0
    total_n_psnr, total_n_ssim = 0, 0

    for i in range(len(input_img)):
        print("Images Processed: %d/ %d  \r" % (i+1, len(input_img)))
        lap_pyramid = Lap_Pyramid_Conv().cuda(2)
        # highmerge = Highmerge().cuda(5)
        #### define model ####
        net = model.Model('hazemodel')
        net.apply(weights_init)
        net.cuda(2)
        net_p = model.Model('hazeproducemodel')
        net_p.apply(weights_init)
        net_p.cuda(2)

        #### define optimizer ####
        optimizer = torch.optim.Adam([
                    {'params': net.parameters()},
                    {'params': net_p.parameters()},],
                    lr=1e-4, betas=(0.99, 0.999), eps=1e-08, weight_decay=1e-2)
        
        #### data process ####
        print('name:', input_img[i])

        Hazy = imageio.imread(args.TestFolderPath+'/hazy/'+input_img[i])
        # SOTS
        ground_truth = imageio.imread(args.TestFolderPath+'/gt/'+input_img[i].split('_')[0]+'.png')
       
        
        # 将图像转换为 Tensor
        Input = _np2Tensor(Hazy)
        Input = (Input/255.).cuda(2)
        print(f"Final Input shape: {Input.shape}")

        ground_truth = _np2Tensor(ground_truth)
        clear = (ground_truth/255.).cuda(2)

         # 调整图像尺寸使其为 32 的倍数
        Hx, Wx = Input.shape[2], Input.shape[3]
        _Hx = Hx - Hx%32
        _Wx = Wx - Wx%32
        Input = Transf(Input, _Hx, _Wx)

        HY, WY = clear.shape[2], clear.shape[3]
        _HY = HY - HY%32
        _WY = WY - WY%32
        ground_truth = Transf(clear, _HY, _WY)

        # 如果图像有 4 个通道，则只保留前三个通道
        if Input.shape[1] == 4:
            Input = torch.cat((Input[0][0].unsqueeze(0).unsqueeze(0), Input[0][1].unsqueeze(0).unsqueeze(0), Input[0][2].unsqueeze(0).unsqueeze(0)), dim=1)
        if ground_truth.shape[1] == 4:
            ground_truth = torch.cat((ground_truth[0][0].unsqueeze(0).unsqueeze(0), ground_truth[0][1].unsqueeze(0).unsqueeze(0), ground_truth[0][2].unsqueeze(0).unsqueeze(0)), dim=1)

        best_psnr = 0
        best_ssim = 0
        #### train ####
        for k in tqdm(range(itr_no), desc="Loading..."):

            #### net and Discriminator start train ####
            net.train()
            net_p.train()
            optimizer.zero_grad()

            #### augment ####数据增强
            Inputmage, gt= _augment(Input, ground_truth)

            

            # # highmerge
            highlist=lap_pyramid.pyramid_decom(Inputmage)
            high=highlist[0]
            

            with autocast():
                trans, atm, HazefreeImage= net(Inputmage,high, 'train')
                # HazefreeImage=highmerge(HazefreeImage1,high)
                highlist2=lap_pyramid.pyramid_decom(HazefreeImage)
                high2=highlist2[0]
                transX, atmX ,HazeProducemage = net_p(HazefreeImage,high2, 'train')
        

        

                #### define Airlight ####
                A = Input.max(dim=3)[0].max(dim=2,keepdim=True)[0].unsqueeze(3)

                #### 上溢 下溢 ####
                otensor = torch.ones(HazefreeImage.shape).cuda(2)
                ztensor = torch.zeros(HazefreeImage.shape).cuda(2)
                lossMx = torch.sum(torch.max(HazefreeImage, otensor))  - torch.sum(otensor)
                lossMn = - torch.sum(torch.min(HazefreeImage, ztensor))
                lossMnx = lossMx + lossMn

                #### cycle loss ####
                loss_cycle = l1_loss(HazeProducemage, Inputmage)


                loss_trans = l1_loss(trans, transX)

                #### Airlight loss ####
                loss_air = l1_loss(atm,A) + l1_loss(atmX,A)

                #### dcp loss ####
                dcp_prior = torch.min(HazefreeImage.permute(0, 2, 3, 1), 3)[0]
                dcp_loss =  l1_loss(dcp_prior, torch.zeros_like(dcp_prior)) - 0.005

                #### total loss ####
                loss = loss_cycle + loss_trans + loss_air + 0.001*lossMnx + 0.001*dcp_loss
                if (k+1) % 200 == 1 or (k+1) % 200 == 0:
                    refinet = t_matting(Inputmage.detach().cpu().numpy(), trans[0].detach().cpu().numpy())
                    J = (Inputmage - (1 - torch.from_numpy(refinet).cuda(2))*atm)/torch.from_numpy(refinet).cuda(2)
                    # J = (Inputmage - (1 - trans) * atm) / trans
                # 计算去雾图像 J 和真实图像 gt 之间的 PSNR 和 SSIM 值
                # 如果当前迭代中的 PSNR 或 SSIM 值比之前记录的最佳值高，则更新 best_psnr 和 best_ssim
                    if psnr(HazefreeImage, gt)>best_psnr:
                        best_psnr = psnr(HazefreeImage, gt)
                    if ssim(HazefreeImage, gt)>best_ssim:
                        best_ssim = ssim(HazefreeImage,gt)
                    print('loss:', loss, 'dcp_loss:', 0.01*dcp_loss, 'current psnr:', psnr(HazefreeImage, gt), psnr(J,gt), 'current ssim:', ssim(HazefreeImage, gt), ssim(J,gt), 'best_psnr:', best_psnr, best_ssim )
                # SOTS
                    torchvision.utils.save_image(torch.cat((HazeProducemage,Inputmage,HazefreeImage,gt),dim=0), args.SavePath+'/'+input_img[i].split('_')[0]+'_H.png')
                # HSTS-synthetic
                # torchvision.utils.save_image(torch.cat((HazeProducemage,Inputmage,HazefreeImage,gt),dim=0), args.SavePath+'/'+input_img[i].split('.')[0]+'_H.png')
            # total_loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        net.eval()
        with torch.no_grad():
            # # highmerge
            highlist=lap_pyramid.pyramid_decom(Input)
            high=highlist[0]
          
            flag='train'
            _trans, _atm, _out = net(Input,high, flag)
            # _out = _out.data[:, :, ori_top:ori_down, ori_left:ori_right]
            _out = torch.clamp(_out, 0, 1)
            flag='test'
            _trans, _atm= net(Input,high,flag)
            # 透射率图（transmission map）的细化（refinement）
            refine_t = t_matting(Input.detach().cpu().numpy(), _trans[0].detach().cpu().numpy())
            _GT = (Input - (1 - torch.from_numpy(refine_t).cuda(2))*_atm)/torch.from_numpy(refine_t).cuda(2)
            # _GT = (Input - (1 - _trans)*_atm)/trans
            _GT = torch.clamp(_GT, 0, 1)
        _out = TransF(_out, Hx, Wx)
        _GT = TransF(_GT, Hx, Wx)
        print("_GT size:", _GT.size())  # Add this
        print("ground_truth:", ground_truth.size())  # And this

        # # _GT 和 ground_truth 之间的 PSNR 和 SSIM 值
        psnr_matting, ssim_matting = psnr(_GT, ground_truth), ssim(_GT, ground_truth)
        total_psnr += psnr_matting
        total_ssim += ssim_matting
        # _out 和 ground_truth 之间的 PSNR 和 SSIM 值
        psnr_normal, ssim_normal = psnr(_out, ground_truth), ssim(_out, ground_truth)
        total_n_psnr += psnr_normal
        total_n_ssim += ssim_normal

        print('保存后的图像PSNR和SSIM:', psnr_normal, psnr_matting, ssim_normal, ssim_matting)
        # print('保存后的图像PSNR和SSIM:', psnr_normal, ssim_normal)

        write_log('./log/HFCM-OHAZE_matting.txt', input_img[i].split('_')[0], psnr_matting,ssim_matting)
        write_log('./log/HFCM-OHAZE_nor.txt', input_img[i].split('_')[0], psnr_normal, ssim_normal)

        # 保存去雾和生成的图像
        torchvision.utils.save_image(ground_truth, args.SavePath+'/'+input_img[i].split('_')[0]+'_gt.png')
        torchvision.utils.save_image(_GT, args.SavePath+'/'+input_img[i].split('_')[0]+'_G.png')
        torchvision.utils.save_image(_out, args.SavePath+'/'+input_img[i].split('_')[0]+'_out.png')
    
    avg_psnr = total_psnr/len(input_img)
    avg_ssim = total_ssim/len(input_img)
    avg_n_psnr = total_n_psnr/len(input_img)
    avg_n_ssim = total_n_ssim/len(input_img)
    write_log('./log/HFCM-OHAZE_matting.txt', 'average', avg_psnr, avg_ssim)
    write_log('./log/HFCM-OHAZE_nor.txt', 'average', avg_n_psnr, avg_n_ssim)
    print('total_psnr:',total_psnr, total_ssim, 'total_n_psnr:',total_n_psnr, total_n_ssim, 'avg_psnr:', avg_psnr, avg_ssim, 'avg_n_psnr:', avg_n_psnr, avg_n_ssim)

test(args) 

