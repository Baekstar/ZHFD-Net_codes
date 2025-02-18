'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-11-27 16:57:21
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-12-13 13:50:39
FilePath: /one_shot/psnr.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import torch
import cv2
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from glob import glob
import torch
import torchvision

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    img1,img2= resize_to_smaller(img1, img2)
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
def resize_to_smaller(img1, img2):
    # Resize or crop both images to the smaller of each dimension
    h = min(img1.size(2), img2.size(2))
    w = min(img1.size(3), img2.size(3))
    return img1[:, :, :h, :w], img2[:, :, :h, :w]

def psnr(imgS, imgG):
    imgS, imgG = resize_to_smaller(imgS, imgG)
    diff = imgS - imgG
    mse = diff.pow(2).mean()
    return -10 * math.log10(mse)

def write_log(file_name, title, psnr, ssim):
    fp = open(file_name, "a+")
    fp.write(title+ ':\n')
    fp.write('PSNR:%0.6f\n'%psnr)
    fp.write('SSIM:%0.6f\n'%ssim)
    fp.close()
def get_file_name(tensor):
    img_path = tensor.item()  # 假设 tensor 是一个包含图像路径的 Tensor
    return os.path.basename(img_path)

if __name__ == '__main__':
    # 图像文件路径
    # gt_folder = '/home/b311/data3/qilishuang/UCL-base/datasets/O-HAZE/testB/'
    # out_folder = '/home/b311/data3/qilishuang/D4-main/checkpoints/test_example_OHAZE/results/reconstruct/'
    # log_file = '/home/b311/data3/qilishuang/D4-main/checkpoints/test_example_OHAZE/results/OHAZE-PSNR.txt'
    # gt_images = sorted(glob(os.path.join(gt_folder, '*.png')))
    # out_images = sorted(glob(os.path.join(out_folder, '*_GT.jpg')))

    # HSTS-HFCM-0.01dcp
    # OHAZE-HFCM HFCM-HSTS-depthdiff-SC-depth2
    gt_folder = '/home/b311/data3/qilishuang/Zero-Shot-HFAM/datasets/NH-HAZE/gt'
    out_folder = '/home/b311/data3/qilishuang/Zero-Shot-HFAM/results/NHHAZE-All'
    log_file = '/home/b311/data3/qilishuang/Zero-Shot-HFAM/best-results/NHHAZE-All.txt'
    output_best_folder='/home/b311/data3/qilishuang/Zero-Shot-HFAM/best-results/NHHAZE-All/'
    gt_images = sorted(glob(os.path.join(gt_folder, '*.png')))
    out_images = sorted(glob(os.path.join(out_folder, '*_G.png')))
    out_images2 = sorted(glob(os.path.join(out_folder, '*_out.png')))

    # 确保输出文件夹存在
    os.makedirs(output_best_folder, exist_ok=True)
    total_psnr = 0
    total_ssim = 0
    num_images = len(gt_images)
    # gt_img_path = './datasets/HSTS/synthetic/gt/0586.jpg'
    # out_img_path = './results/HSTS/synthetic/0586.jpg_G.png'

    for gt_img_path, out_img_path,out_img_path2 in zip(gt_images, out_images,out_images2):
        fn_gt = torch.from_numpy(cv2.imread(gt_img_path).astype('float32')/255).permute(2,0,1).unsqueeze(0)
        fn_out = torch.from_numpy(cv2.imread(out_img_path).astype('float32')/255).permute(2,0,1).unsqueeze(0)
        fn_out2 = torch.from_numpy(cv2.imread(out_img_path2).astype('float32')/255).permute(2,0,1).unsqueeze(0)

    
    
        Hx, Wx = fn_gt.shape[2], fn_gt.shape[3]
        Hx = Hx - Hx%32
        Wx = Wx - Wx%32
        fn_gt = fn_gt[:,:,0:Hx, 0:Wx]

        # print('shape:', fn_gt.shape, fn_out.shape)

        ps = psnr(fn_out, fn_gt)
        ss = ssim(fn_out, fn_gt)
        ps2 = psnr(fn_out2, fn_gt)
        ss2 = ssim(fn_out2, fn_gt)
        if ps2>ps:
            print(f'{os.path.basename(gt_img_path)} - PSNR: {ps2}, SSIM: {ss2}')
            total_psnr += ps2
            total_ssim += ss2


            file_name = os.path.basename(gt_img_path)
            write_log(log_file, file_name, ps2, ss2)
            # 读取图像，确保按原始 BGR 读取，并保持原始范围
            img_out2 = cv2.imread(out_img_path2, cv2.IMREAD_UNCHANGED)  # 读取原始图像
            img_out2_rgb = cv2.cvtColor(img_out2, cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB

            # 将图像转换为 Tensor，并保持像素值在 [0, 1] 范围内
            img_out2_tensor = torch.from_numpy(img_out2_rgb.astype('float32') / 255).permute(2, 0, 1)

            torchvision.utils.save_image(img_out2_tensor, os.path.join(output_best_folder, os.path.basename(out_img_path2)))
        else:
            print(f'{os.path.basename(gt_img_path)} - PSNR: {ps}, SSIM: {ss}')
        
            total_psnr += ps
            total_ssim += ss


            file_name = os.path.basename(gt_img_path)
            write_log(log_file, file_name, ps, ss)
            # 读取图像，确保按原始 BGR 读取，并保持原始范围
            img_out = cv2.imread(out_img_path, cv2.IMREAD_UNCHANGED)  # 读取原始图像
            img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB

            # 将图像转换为 Tensor，并保持像素值在 [0, 1] 范围内
            img_out_tensor = torch.from_numpy(img_out_rgb.astype('float32') / 255).permute(2, 0, 1)
            torchvision.utils.save_image(img_out_tensor, os.path.join(output_best_folder, os.path.basename(out_img_path)))

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images

    print(f'Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}')
    with open(log_file, "a+") as fp:
        fp.write(f'Average PSNR: {avg_psnr:.6f}\n')
        fp.write(f'Average SSIM: {avg_ssim:.6f}\n')
    

