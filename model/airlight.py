# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# def load_image(image_path):
#     # Load image and convert it to grayscale
#     image = Image.open(image_path).convert("L")
#     transform = transforms.ToTensor()
#     return transform(image).unsqueeze(0)  # Convert to 4D tensor for batch processing

# def get_average_gray_score(image):
#     # Calculate the average of gray values in the tensor
#     return torch.mean(image)

# def quadtree_split(image, depth=0, max_depth=3):
#     _, _, height, width = image.shape
#     if depth == max_depth or min(height, width) <= 2:
#         return [image]
    
#     mid_height, mid_width = height // 2, width // 2
#     top_left = image[:, :, :mid_height, :mid_width]
#     top_right = image[:, :, :mid_height, mid_width:]
#     bottom_left = image[:, :, mid_height:, :mid_width]
#     bottom_right = image[:, :, mid_height:, mid_width:]

#     return (quadtree_split(top_left, depth+1, max_depth) +
#             quadtree_split(top_right, depth+1, max_depth) +
#             quadtree_split(bottom_left, depth+1, max_depth) +
#             quadtree_split(bottom_right, depth+1, max_depth))

# def select_and_divide(image, e=1.1, MT=1, w=25):
#     regions = quadtree_split(image)
#     scores = [get_average_gray_score(region) for region in regions]
#     top_scores = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]  # Get indices of two highest scores

#     # Weighting factor and selection
#     if top_scores[0] < 2 and top_scores[1] < 2:  # Both in the upper half
#         weighted_scores = [e * scores[i] for i in top_scores] + [scores[i] for i in range(2, 4)]
#     else:
#         weighted_scores = scores

#     selected_index = max(range(len(weighted_scores)), key=lambda i: weighted_scores[i])
#     selected_region = regions[selected_index]
#     _, _, selected_height, selected_width = selected_region.shape

#     # Termination conditions
#     if selected_width < w or abs(scores[top_scores[0]] - scores[top_scores[1]]) < MT:
#         return torch.mean(selected_region, dim=[2, 3])  # Return the atmospheric light as the average of the selected region
#     else:
#         return select_and_divide(selected_region, e, MT, w)  # Recursive call

# # Example usage
# image_path = '/home/b311/data3/qilishuang/Zero-Shot/datasets/HSTS/synthetic/hazy/0586.jpg'
# image = load_image(image_path)
# # A = select_and_divide(image)
# A = image.max(dim=3)[0].max(dim=2,keepdim=True)[0].unsqueeze(3)
# #print("Estimated Atmospheric Light:", A)


import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image
from PIL import Image, ImageEnhance


def adjust_saturation(tensor_image, saturation_factor):
    pil_image = TF.to_pil_image(tensor_image)
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(saturation_factor)
    return TF.to_tensor(pil_image)

def adjust_sharpness(tensor_image, sharpness_factor):
    pil_image = TF.to_pil_image(tensor_image)
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(sharpness_factor)
    return TF.to_tensor(pil_image)

def adjust_brightness(tensor_image, brightness_factor):
    pil_image = TF.to_pil_image(tensor_image)
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness_factor)
    return TF.to_tensor(pil_image)

import numpy as np

def is_blue_pixel(pixel, threshold=50):
    r, g, b = pixel
    return b > r + threshold and b > g + threshold and b>100

def blue_ratio_in_image(tensor_image, threshold=100):
    pil_image = TF.to_pil_image(tensor_image)
    image_pixels = np.array(pil_image)
    blue_pixels = sum(is_blue_pixel(pixel, threshold) for row in image_pixels for pixel in row)
    
    total_pixels = pil_image.size[0] * pil_image.size[1]
    blue_ratio = blue_pixels / total_pixels
    return blue_ratio



def process_image(image, sharpness_factor=1.5,brightness_factor=1.1,saturation_factor=0.9): 
    image = adjust_sharpness(image, sharpness_factor)
    image =  adjust_brightness(image, brightness_factor)
    # # 调整饱和度
    image = adjust_saturation(image, saturation_factor)
    
    return image
# import os
# # 输入和输出文件夹路径
# input_folder = '/home/b311/data3/qilishuang/Zero-Shot/results/OHAZE-HFCM'
# output_folder = '/home/b311/data3/qilishuang/Zero-Shot/results/OHAZE-HFCM-gt'
# if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
# for filename in os.listdir(input_folder):
#         if filename.endswith(".jpg") or filename.endswith("_G.png"):
#             image_path = os.path.join(input_folder, filename)
#             image = Image.open(image_path)
            

#             threshold=50
#             blue_threshold=0
#             # 计算蓝色区域的比例
#             blue_ratio = blue_ratio_in_image(image, threshold)
#             print(f"Blue ratio in {filename}: {blue_ratio:.2f}")
            
#             # 如果蓝色区域占比超过阈值，则处理图片
#             if blue_ratio > blue_threshold:

#                 # 处理图片
#                 final_image = process_image(image,sharpness_factor=1.5,brightness_factor=1.1,saturation_factor=0.9)
#             else:
#                 final_image =image

#             # 保存图片到指定路径
#             save_path = os.path.join(output_folder, filename)
#             # 转换 PIL 图像为 Tensor
#             transform = transforms.ToTensor()
#             tensor_image = transform(final_image)
    
#             # 保存 Tensor 图像
#             save_image(tensor_image, save_path)

# print(f"Image saved to ")
