import torch
import torchvision
import imageio
import model
import argparse
import os
from psnr import ssim
from cv2.ximgproc import guidedFilter
from utils.utils import _np2Tensor, _augment, psnr, t_matting, write_log,write_log_JC
import torch.nn as nn
import torchvision.transforms as trans
from tqdm import tqdm
import math
import torchvision.transforms.functional as TF
from torchvision import transforms
from model.highmerge import Lap_Pyramid_Conv,padding_image,resize_and_pad_image
import numpy as np
from PIL import Image
from model.Loss import SSIMLoss,CustomLoss
from model.CR import ContrastLoss
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from model.airlight import process_image,blue_ratio_in_image
from model.depthimg import DN
from PIL import Image, ImageFilter
from inference_single_image import inference
# 
def resize_width(image, target_w):
    h, w = image.shape[2], image.shape[3]
    scale = target_w / w
    new_h, new_w = int(h * scale), target_w
    resized_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return resized_image

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
parser.add_argument('--TestFolderPath', type=str, default='/home/b311/data3/qilishuang/Zero-Shot-HFAM/datasets/HSTS/', help='Hazy Image folder name') 
parser.add_argument('--SavePath', type=str, default='./results/HSTS', help='SavePath Name')
args = parser.parse_args()

itr_no = 10000
l1_loss = nn.MSELoss()
# customloss=CustomLoss()

scaler = GradScaler()
def test(args):
    input_img = os.listdir(args.TestFolderPath+'/hazy/')
    input_img.sort()
    os.makedirs(args.SavePath+'/', exist_ok=True)
    
    total_psnr, total_ssim = 0, 0
    total_n_psnr, total_n_ssim = 0, 0

    for i in range(len(input_img)):
        print("Images Processed: %d/ %d  \r" % (i+1, len(input_img)))
        lap_pyramid = Lap_Pyramid_Conv().cuda(5)
        # highmerge = Highmerge().cuda(5)
        #### define model ####
        net = model.Model('hazemodel')
        net.apply(weights_init)
        net.cuda(5)
        net_p = model.Model('hazeproducemodel')
        net_p.apply(weights_init)
        net_p.cuda(5)
        depthmap=DN()
        depthmap.cuda(5)


        #### define optimizer ####
        optimizer = torch.optim.Adam([
                    {'params': net.parameters()},
                    {'params': net_p.parameters()},
                    {'params': depthmap.parameters()},],
                    lr=1e-4, betas=(0.99, 0.999), eps=1e-08, weight_decay=1e-2)
        
        #### data process ####
        print('name:', input_img[i])

        Hazy = imageio.imread(args.TestFolderPath+'/hazy/'+input_img[i])
        
        # SOTS
        # ground_truth = imageio.imread(args.TestFolderPath+'/gt/'+input_img[i].split('_')[0]+'_GT'+'.png')
        # HSTS
        ground_truth = imageio.imread(args.TestFolderPath+'/gt/'+input_img[i])
        
        # 将图像转换为 Tensor
        Input = _np2Tensor(Hazy)
        Input = (Input/255.).cuda(5)
        print(f"Final Input shape: {Input.shape}")

        ground_truth = _np2Tensor(ground_truth)
        clear = (ground_truth/255.).cuda(5)

        # # 定义 transforms
        # transform = transforms.Compose([
        #     transforms.Resize((256, 256)),  # 调整到 256x256 分辨率
        #     transforms.ToTensor()  # 转换为 Tensor，并归一化到 [0, 1]
        # ])
        

         # 调整图像尺寸使其为 32 的倍数
        Hx, Wx = Input.shape[2], Input.shape[3]
        _Hx = Hx - Hx%32
        _Wx = Wx - Wx%32
        Input = Transf(Input, _Hx, _Wx)
        # Input = Transf(Input, 768, 768)

        HY, WY = clear.shape[2], clear.shape[3]
        _HY = HY - HY%32
        _WY = WY - WY%32
        ground_truth = Transf(clear, _HY, _WY)
        # ground_truth = Transf(clear, 768, 768)

        # 如果图像有 4 个通道，则只保留前三个通道
        if Input.shape[1] == 4:
            Input = torch.cat((Input[0][0].unsqueeze(0).unsqueeze(0), Input[0][1].unsqueeze(0).unsqueeze(0), Input[0][2].unsqueeze(0).unsqueeze(0)), dim=1)
        if ground_truth.shape[1] == 4:
            ground_truth = torch.cat((ground_truth[0][0].unsqueeze(0).unsqueeze(0), ground_truth[0][1].unsqueeze(0).unsqueeze(0), ground_truth[0][2].unsqueeze(0).unsqueeze(0)), dim=1)




        best_psnr = 0
        best_ssim = 0
        best_model_path='./best_model/HSTS.pth'
        #### train ####
        for k in tqdm(range(itr_no), desc="Loading..."):

            #### net and Discriminator start train ####
            net.train()
            net_p.train()
            depthmap.train()
            optimizer.zero_grad()

            #### augment ####数据增强
            Inputmage, gt= _augment(Input, ground_truth)


            #### two ways removal or produce fog ####去雾和生成雾两种路径
            

            # # highmerge
            highlist=lap_pyramid.pyramid_decom(Inputmage)
            high=highlist[0]
            

            with autocast():
                trans, atm, HazefreeImage= net(Inputmage,high, 'train')
                # HazefreeImage=highmerge(HazefreeImage1,high)
                highlist2=lap_pyramid.pyramid_decom(HazefreeImage)
                high2=highlist2[0]
                transX, atmX ,HazeProducemage = net_p(HazefreeImage,high2, 'train')
        
                # depth_map
                model_path = "./MODEL_PATH"
                hazy_depth=inference(Inputmage,model_path).cuda(5)
                rehaze_depth=inference(HazeProducemage,model_path).cuda(5)

                # loss_depth=l1_loss(hazy_depth,rehaze_depth)

                diff_dehaze = torch.sub(HazeProducemage, Inputmage)
                B, C, H, W = diff_dehaze.shape
                diff_dehaze = diff_dehaze.permute(0, 2, 3, 1)
                diff_dehaze = diff_dehaze.reshape(-1, C * H * W)
                epsilon = 1e-7
                diff_d_w = F.softmax(diff_dehaze, dim=-1) + epsilon
                diff_d_w = diff_d_w.reshape(B, H, W, C).permute(0, 3, 1, 2)
                diff_dehaze_w = torch.sum(diff_d_w, dim=1, keepdim=True)
                # 深度图
                weighted_depth_output_img = torch.mul(rehaze_depth, diff_dehaze_w)
                weighted_real_img_2_depth_img = torch.mul(hazy_depth, diff_dehaze_w)
                criterion_depth = nn.L1Loss()
                loss_depth_consis = criterion_depth(weighted_depth_output_img, weighted_real_img_2_depth_img)
                loss_depth_consis_w = criterion_depth(rehaze_depth, hazy_depth)
                loss_depth = loss_depth_consis + loss_depth_consis_w
        

                #### define Airlight ####
                A = Input.max(dim=3)[0].max(dim=2,keepdim=True)[0].unsqueeze(3)
                # write_log_JC('./log/HSTS.txt',HazefreeImage)

                #### 上溢 下溢 ####
                otensor = torch.ones(HazefreeImage.shape).cuda(5)
                ztensor = torch.zeros(HazefreeImage.shape).cuda(5)
                lossMx = torch.sum(torch.max(HazefreeImage, otensor))  - torch.sum(otensor)
                lossMn = - torch.sum(torch.min(HazefreeImage, ztensor))
                lossMnx = lossMx + lossMn

                #### cycle loss ####
                loss_cycle = l1_loss(HazeProducemage, Inputmage)
                # loss_cycle = customloss(HazeProducemage, Inputmage)

                loss_trans = l1_loss(trans, transX)

                #### Airlight loss ####
                loss_air = l1_loss(atm,A) + l1_loss(atmX,A)

                #### dcp loss ####
                dcp_prior = torch.min(HazefreeImage.permute(0, 2, 3, 1), 3)[0]
                dcp_loss =  l1_loss(dcp_prior, torch.zeros_like(dcp_prior)) - 0.005

                #### total loss ####
                loss = loss_cycle + loss_trans + loss_air + 0.001*lossMnx + 0.01*dcp_loss +loss_depth
                # loss = loss_cycle + loss_trans + loss_air  + 0.01*dcp_loss +loss_depth
                # write_log_JC('./log/LOPP-JC2.txt',HazefreeImage)
                if (k+1) % 200 == 1 or (k+1) % 200 == 0:
                    # print(f"HazefreeImage shape: {HazefreeImage.shape}")
                    # refinet = t_matting(Inputmage.detach().cpu().numpy(),high.detach().cpu().numpy(), trans[0].detach().cpu().numpy())
                    # J = (Inputmage - (1 - trans) * atm) / trans
                    
    
                    refinet = t_matting(Inputmage.detach().cpu().numpy(), trans[0].detach().cpu().numpy())
                    J = (Inputmage - (1 - torch.from_numpy(refinet).cuda(5))*atm)/torch.from_numpy(refinet).cuda(5)
                        # HazefreeImage =HazefreeImage
                    # refinet = t_matting(Inputmage.detach().cpu().numpy(),high.detach().cpu().numpy(), trans[0].detach().cpu().numpy())
                    #J = (Inputmage - (1 - torch.from_numpy(trans).cuda(5))*atm)/torch.from_numpy(trans).cuda(5)
                    # J = (Inputmage - (1 - trans) * atm) / trans
                # 计算去雾图像 J 和真实图像 gt 之间的 PSNR 和 SSIM 值
                # 如果当前迭代中的 PSNR 或 SSIM 值比之前记录的最佳值高，则更新 best_psnr 和 best_ssim
                    # if psnr(HazefreeImage, gt)>best_psnr:
                    #     best_psnr = psnr(HazefreeImage, gt)
                    # if ssim(HazefreeImage, gt)>best_ssim:
                    #     best_ssim = ssim(HazefreeImage,gt)
                    if psnr(HazefreeImage, gt)>best_psnr:
                        best_psnr = psnr(HazefreeImage, gt)
                        best_ssim = ssim(HazefreeImage, gt)
                        torch.save({
                            'net_state_dict': net.state_dict(),
                            'net_p_state_dict': net_p.state_dict(),
                            # 'depthmap_state_dict': depthmap.state_dict(),
                            }, best_model_path)
                    print('loss:', loss, 'dcp_loss:', 0.01*dcp_loss, 'current psnr:', psnr(HazefreeImage, gt), psnr(J,gt), 'current ssim:', ssim(HazefreeImage, gt), ssim(J,gt), 'best_psnr:', best_psnr, best_ssim )
                # SOTS
                    torchvision.utils.save_image(torch.cat((HazeProducemage,Inputmage,HazefreeImage,gt),dim=0), args.SavePath+'/'+input_img[i].split('.')[0]+'_H.png')
                    torchvision.utils.save_image(hazy_depth, args.SavePath+'/'+input_img[i].split('.')[0]+'_depth_hazy.png')
                    torchvision.utils.save_image(rehaze_depth, args.SavePath+'/'+input_img[i].split('.')[0]+'_depth_rehaze.png')
                    # write_log_loss('./log/loss_depth-SC-estCA.txt',loss,dcp_loss, loss_cycle,loss_trans,loss_air,loss_depth)
                    # torchvision.utils.save_image(torch.cat((hazy_depth,dehaze_depth),dim=0), args.SavePath+'/'+input_img[i].split('_')[0]+'_depth.png')
                # HSTS-synthetic
                # torchvision.utils.save_image(torch.cat((HazeProducemage,Inputmage,HazefreeImage,gt),dim=0), args.SavePath+'/'+input_img[i].split('.')[0]+'_H.png')
            # total_loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


        # 训练结束后，加载最佳模型参数并用于测试
        best_checkpoint = torch.load(best_model_path)
        net.load_state_dict(best_checkpoint['net_state_dict'])
        net.eval()
        with torch.no_grad():
            # # highmerge
            highlist=lap_pyramid.pyramid_decom(Input)
            high=highlist[0]
           
            flag='train'
            _trans, _atm, _out = net(Input,high, flag)
            ## 透射率图（transmission map）的细化（refinement）
            #refine_t1 = t_matting(Input.detach().cpu().numpy(),high.detach().cpu().numpy(), _trans[0].detach().cpu().numpy())
            # _out = (Input - (1 - torch.from_numpy(_trans).cuda(5))*_atm)/torch.from_numpy(_trans).cuda(5)
            # _out= (Input - (1 - _trans)*_atm)/_trans
            _out = torch.clamp(_out, 0, 1)

            flag='test'
            _trans, _atm= net(Input,high,flag)
            # 透射率图（transmission map）的细化（refinement）
            # _GT = (Input - (1 - _trans)*_atm)/_trans
            refine_t = t_matting(Input.detach().cpu().numpy(), _trans[0].detach().cpu().numpy())
            _GT = (Input - (1 - torch.from_numpy(refine_t).cuda(5))*_atm)/torch.from_numpy(refine_t).cuda(5)
           
            _GT = torch.clamp(_GT, 0, 1)
        
        
        _out = TransF(_out, Hx, Wx)
        _GT = TransF(_GT, Hx, Wx)
        print("_GT size:", _GT.size())  # Add this
        print("ground_truth:", ground_truth.size())  # And this#

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

        write_log('./log/HSTS_matting.txt', input_img[i].split('.')[0], psnr_matting,ssim_matting)
        write_log('./log/HSTS_nor.txt', input_img[i].split('.')[0], psnr_normal, ssim_normal)

        # 保存去雾和生成的图像
        # torchvision.utils.save_image(ground_truth, args.SavePath+'/'+input_img[i].split('_')[0]+'_gt.png')
        # torchvision.utils.save_image(Input, args.SavePath+'/'+input_img[i].split('_')[0]+'_hazy.png')
        torchvision.utils.save_image(_GT, args.SavePath+'/'+input_img[i].split('.')[0]+'_G.png')
        torchvision.utils.save_image(_out, args.SavePath+'/'+input_img[i].split('.')[0]+'_out.png')
    
    avg_psnr = total_psnr/len(input_img)
    avg_ssim = total_ssim/len(input_img)
    avg_n_psnr = total_n_psnr/len(input_img)
    avg_n_ssim = total_n_ssim/len(input_img)
    write_log('./log/HSTS_matting.txt', 'average', avg_psnr, avg_ssim)
    write_log('./log/HSTS_nor.txt', 'average', avg_n_psnr, avg_n_ssim)
    print('total_psnr:',total_psnr, total_ssim, 'total_n_psnr:',total_n_psnr, total_n_ssim, 'avg_psnr:', avg_psnr, avg_ssim, 'avg_n_psnr:', avg_n_psnr, avg_n_ssim)

test(args) 

