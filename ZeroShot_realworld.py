import torch
import torchvision
import imageio
import model
import argparse
import os
from psnr import ssim
from cv2.ximgproc import guidedFilter
from utils.utils import _np2Tensor, _augment, psnr, t_matting, write_log,write_log_loss
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
from inference_single_image import inference

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
parser.add_argument('--TestFolderPath', type=str, default='/home/b311/data3/qilishuang/HSTS/real-world/', help='Hazy Image folder name') 
parser.add_argument('--SavePath', type=str, default='./results/realworld-all', help='SavePath Name')
args = parser.parse_args()

itr_no = 10000
l1_loss = nn.MSELoss()
# customloss=CustomLoss()

scaler = GradScaler()
def test(args):
    input_img = os.listdir(args.TestFolderPath)
    input_img.sort()
    os.makedirs(args.SavePath+'/', exist_ok=True)
    
    # total_psnr, total_ssim = 0, 0
    # total_n_psnr, total_n_ssim = 0, 0

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
        # depthmap=DN()
        # depthmap.cuda(5)


        #### define optimizer ####
        optimizer = torch.optim.Adam([
                    {'params': net.parameters()},
                    {'params': net_p.parameters()},],
                    # {'params': depthmap.parameters()},
                    lr=1e-4, betas=(0.99, 0.999), eps=1e-08, weight_decay=1e-2)
        
        #### data process ####
        print('name:', input_img[i])

        Hazy = imageio.imread(args.TestFolderPath+input_img[i])
        # SOTS/OHAZE
        # ground_truth = imageio.imread(args.TestFolderPath+'/gt/'+input_img[i].split('_')[0]+'.png')
        # HSTS-synthetic
        # ground_truth = imageio.imread(args.TestFolderPath+'/gt/'+input_img[i])
        
        # 将 numpy 数组类型的图像转换为 PIL 图像
        # Hazy_pil = Image.fromarray(Hazy)
        # ground_truth_pil = Image.fromarray(ground_truth)
        # ii, j, h, w = transforms.RandomCrop.get_params(Hazy_pil, output_size=(256, 256))
        # Hazy = TF.crop(Hazy_pil, ii, j, h, w)
        # ground_truth = TF.crop(ground_truth_pil, ii, j, h, w)
        # # 将裁剪后的 PIL 图像转换回 numpy 数组
        # Hazy = np.array(Hazy)
        # ground_truth = np.array(ground_truth)


        # # Convert to PyTorch tensor and check the shape
        # if Hazy.ndim == 2:  # if image is grayscale, convert to RGB by repeating the channels
        #     Hazy = np.repeat(Hazy[:, :, np.newaxis], 3, axis=2)
        # elif Hazy.ndim == 3 and Hazy.shape[2] == 4:  # if image has alpha channel, remove it
        #     Hazy = Hazy[:, :, :3]
        # print(f"Before conversion: type={type(Hazy)}, shape={Hazy.shape}")
        # Hazy = torch.from_numpy(Hazy).permute(2, 0, 1).unsqueeze(0).float().cuda(5)  # 转换为 (1, channels, height, width)
        # print(f"After conversion: type={type(Hazy)}, shape={Hazy.shape}")
        # # Convert to PyTorch tensor and check the shape
        # if ground_truth.ndim == 2:  # if image is grayscale, convert to RGB by repeating the channels
        #     ground_truth = np.repeat(ground_truth[:, :, np.newaxis], 3, axis=2)
        # elif ground_truth.ndim == 3 and ground_truth.shape[2] == 4:  # if image has alpha channel, remove it
        #     ground_truth = ground_truth[:, :, :3]
        # ground_truth = torch.from_numpy(ground_truth).permute(2, 0, 1).unsqueeze(0).float().cuda(5)  # 转换为 (1, channels, height, width)
        # print(f"Hazy shape: {Hazy.shape}")
        # h, w = Hazy.shape[2], Hazy.shape[3]
        # max_h = int(math.ceil(h / 256)) * 256
        # max_w = int(math.ceil(w / 256)) * 256

        # Hazy, ori_left, ori_right, ori_top, ori_down = resize_and_pad_image(Hazy, max_h, max_w)
        # ground_truth, ori_left, ori_right, ori_top, ori_down = resize_and_pad_image(ground_truth, max_h, max_w)
        # print(f"Hazy shape: {Hazy.shape}")
        # print(f"Padding: left={ori_left}, right={ori_right}, top={ori_top}, down={ori_down}")
        
        # # 缩小图像尺寸
        # scale_factor = 256  # 缩小一半
        # Hazy = resize_width(Hazy, scale_factor)
        # ground_truth = resize_width(ground_truth, scale_factor)
        # print(f"Resized Hazy shape: {Hazy.shape}")
        # print(f"Resized ground_truth shape: {ground_truth.shape}")
        
        # 将图像转换为 Tensor
        Input = _np2Tensor(Hazy)
        Input = (Input/255.).cuda(5)
        print(f"Final Input shape: {Input.shape}")

        # ground_truth = _np2Tensor(ground_truth)
        # clear = (ground_truth/255.).cuda(5)

         # 调整图像尺寸使其为 32 的倍数
        Hx, Wx = Input.shape[2], Input.shape[3]
        _Hx = Hx - Hx%32
        _Wx = Wx - Wx%32
        Input = Transf(Input, 768, 768)

        # HY, WY = clear.shape[2], clear.shape[3]
        # _HY = HY - HY%32
        # _WY = WY - WY%32
        # ground_truth = Transf(clear, _HY, _WY)

        # 如果图像有 4 个通道，则只保留前三个通道
        if Input.shape[1] == 4:
            Input = torch.cat((Input[0][0].unsqueeze(0).unsqueeze(0), Input[0][1].unsqueeze(0).unsqueeze(0), Input[0][2].unsqueeze(0).unsqueeze(0)), dim=1)
        # if ground_truth.shape[1] == 4:
        #     ground_truth = torch.cat((ground_truth[0][0].unsqueeze(0).unsqueeze(0), ground_truth[0][1].unsqueeze(0).unsqueeze(0), ground_truth[0][2].unsqueeze(0).unsqueeze(0)), dim=1)

        best_psnr = 0
        best_ssim = 0
        # best_model_path='./best_model/All-real.pth'
        #### train ####
        for k in tqdm(range(itr_no), desc="Loading..."):

            #### net and Discriminator start train ####
            net.train()
            net_p.train()
            # depthmap.train()
            optimizer.zero_grad()

            #### augment ####数据增强
            Inputmage, gt= _augment(Input, Input)


            #### two ways removal or produce fog ####去雾和生成雾两种路径
            

            # # highmerge
            highlist=lap_pyramid.pyramid_decom(Inputmage)
            high=highlist[0]
            # h, w = Inputmage.shape[2], Inputmage.shape[3]
            # max_h = int(math.ceil(h / 256)) * 256
            # max_w = int(math.ceil(w / 256)) * 256
            # Inputmage, ori_left, ori_right, ori_top, ori_down = padding_image(Inputmage, max_h, max_w)
            # high, _, _, _, _ = padding_image(high, max_h, max_w)

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
                if (k+1) % 200 == 1 or (k+1) % 200 == 0:
                    # print(f"HazefreeImage shape: {HazefreeImage.shape}")
                    # refinet = t_matting(Inputmage.detach().cpu().numpy(),high.detach().cpu().numpy(), trans[0].detach().cpu().numpy())
                    # J = (Inputmage - (1 - trans) * atm) / trans
                    
                    # threshold=50
                    # blue_threshold=0
                    # # 计算蓝色区域的比例
                    # blue_ratio = blue_ratio_in_image(J.squeeze(0), threshold)
                    # # 如果蓝色区域占比超过阈值，则处理图片
                    # if blue_ratio > blue_threshold:
                        # 处理图片
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
                    # if psnr(HazefreeImage, gt)>best_psnr:
                    #     best_psnr = psnr(HazefreeImage, gt)
                    #     best_ssim = ssim(HazefreeImage, gt)
                    #     torch.save({
                    #         'net_state_dict': net.state_dict(),
                    #         'net_p_state_dict': net_p.state_dict(),
                    #         # 'depthmap_state_dict': depthmap.state_dict(),
                    #         }, best_model_path)
                    # print('loss:', loss, 'dcp_loss:', 0.01*dcp_loss, 'current psnr:', psnr(HazefreeImage, gt), psnr(J,gt), 'current ssim:', ssim(HazefreeImage, gt), ssim(J,gt), 'best_psnr:', best_psnr, best_ssim )
                # SOTS
                    torchvision.utils.save_image(torch.cat((HazeProducemage,Inputmage,HazefreeImage),dim=0), args.SavePath+'/'+input_img[i].split('.')[0]+'_H.png')
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


        # # 训练结束后，加载最佳模型参数并用于测试
        # best_checkpoint = torch.load(best_model_path)
        # net.load_state_dict(best_checkpoint['net_state_dict'])
        net.eval()
        with torch.no_grad():
            # # highmerge
            highlist=lap_pyramid.pyramid_decom(Input)
            high=highlist[0]
            # h, w = Input.shape[2], Input.shape[3]
            # max_h = int(math.ceil(h / 256)) * 256
            # max_w = int(math.ceil(w / 256)) * 256
            # Input, ori_left, ori_right, ori_top, ori_down = padding_image(Input, max_h, max_w)
            # high, _, _, _, _ = padding_image(high, max_h, max_w)
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
            # threshold=50
            # blue_threshold=0.001
            # # 计算蓝色区域的比例
            # blue_ratio = blue_ratio_in_image(_GT.squeeze(0), threshold)
            # # 如果蓝色区域占比超过阈值，则处理图片
            # if blue_ratio > blue_threshold:
            #     # 处理图片
            #     refine_t = t_matting(Input.detach().cpu().numpy(), _trans[0].detach().cpu().numpy())
            #     _GT = (Input - (1 - torch.from_numpy(refine_t).cuda(5))*_atm)/torch.from_numpy(refine_t).cuda(5)
            
            #_GT = (Input - (1 - torch.from_numpy(_trans).cuda(5))*_atm)/torch.from_numpy(_trans).cuda(5)
            # _GT = (Input - (1 - _trans)*_atm)/_trans
            # _out = torch.clamp(_out, 0, 1)
            _GT = torch.clamp(_GT, 0, 1)
        _out = TransF(_out, Hx, Wx)
        _GT = TransF(_GT, Hx, Wx)
        print("_GT size:", _GT.size())  # Add this
        # print("ground_truth:", ground_truth.size())  # And this#

        # # _GT 和 ground_truth 之间的 PSNR 和 SSIM 值
        # psnr_matting, ssim_matting = psnr(_GT, ground_truth), ssim(_GT, ground_truth)
        # total_psnr += psnr_matting
        # total_ssim += ssim_matting
        # # _out 和 ground_truth 之间的 PSNR 和 SSIM 值
        # psnr_normal, ssim_normal = psnr(_out, ground_truth), ssim(_out, ground_truth)
        # total_n_psnr += psnr_normal
        # total_n_ssim += ssim_normal

        # print('保存后的图像PSNR和SSIM:', psnr_normal, psnr_matting, ssim_normal, ssim_matting)
        # # print('保存后的图像PSNR和SSIM:', psnr_normal, ssim_normal)

        # write_log('./log/All-bestmodel-Ada_matting.txt', input_img[i].split('_')[0], psnr_matting,ssim_matting)
        # write_log('./log/All-bestmodel-Ada_nor.txt', input_img[i].split('_')[0], psnr_normal, ssim_normal)

        # 保存去雾和生成的图像
        # torchvision.utils.save_image(ground_truth, args.SavePath+'/'+input_img[i].split('_')[0]+'_gt.png')
        # torchvision.utils.save_image(Input, args.SavePath+'/'+input_img[i].split('_')[0]+'_hazy.png')
        torchvision.utils.save_image(_GT, args.SavePath+'/'+input_img[i].split('.')[0]+'_G.png')
        torchvision.utils.save_image(_out, args.SavePath+'/'+input_img[i].split('.')[0]+'_out.png')
    
    # avg_psnr = total_psnr/len(input_img)
    # avg_ssim = total_ssim/len(input_img)
    # avg_n_psnr = total_n_psnr/len(input_img)
    # avg_n_ssim = total_n_ssim/len(input_img)
    # write_log('./log/All-bestmodel-Ada_matting.txt', 'average', avg_psnr, avg_ssim)
    # write_log('./log/All-bestmodel-Ada_nor.txt', 'average', avg_n_psnr, avg_n_ssim)
    # print('total_psnr:',total_psnr, total_ssim, 'total_n_psnr:',total_n_psnr, total_n_ssim, 'avg_psnr:', avg_psnr, avg_ssim, 'avg_n_psnr:', avg_n_psnr, avg_n_ssim)

test(args) 

