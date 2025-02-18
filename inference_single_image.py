# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import torchvision
import os
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

import networks
from layers import disp_to_depth


# def parse_args():
#     parser = argparse.ArgumentParser(description='Inference on one Single Image.')
#     parser.add_argument('--image_path', type=str,
#                         help='path to a test image',
#                         required=True)
#     parser.add_argument("--load_weights_folder",
#                         type=str,
#                         help="name of model to load",
#                         required=True)
#     parser.add_argument("--no_cuda",
#                         help='if set, disables CUDA',
#                         action='store_true')

#     return parser.parse_args()

def prepare_model_for_test(device,model_path):
    # model_path = args.load_weights_folder
    # print("-> Lo/ading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    decoder_path = os.path.join(model_path, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    decoder_dict = torch.load(decoder_path, map_location=device)

    encoder = networks.ResnetEncoder(18, False)
    decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, 
        scales=range(1),
    )

    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    decoder.load_state_dict(decoder_dict)
    
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    
    return encoder, decoder, encoder_dict['height'], encoder_dict['width']



def inference(input_image,model_path):
    if torch.cuda.is_available():
        device = torch.device("cuda:5")
    else:
        device = torch.device("cpu")
    
    encoder, decoder, thisH, thisW = prepare_model_for_test(device,model_path)
    # print("thisH:",thisH)
    # print("thisW:",thisW)
    # image_path = args.image_path
    # print("-> Inferencing on image ", image_path)

    with torch.no_grad():
        # Load image and preprocess
        # input_image = pil.open(image_path).convert('RGB')
        # extension = image_path.split('.')[-1]
        original_height,original_width  =  input_image.shape[2], input_image.shape[3]
        # print("thisW:",thisW)
        # print("thisW:",thisW)
        # input_image = input_image.resize((thisH, thisW), pil.LANCZOS)
        # input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        if input_image.shape[2:] != (thisH, thisW):
            input_image = torch.nn.functional.interpolate(
                input_image, size=(thisH, thisW), mode='bilinear', align_corners=False)

        # PREDICTION
        input_image = input_image.to(device)
        outputs = decoder(encoder(input_image))

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving numpy file
        # name_dest_npy = image_path.replace('./_depth.npy') 
        # print("-> Saving depth npy to ", name_dest_npy)
        # scaled_disp, _ = disp_to_depth(disp, 0.1, 10)
        # np.save(name_dest_npy, scaled_disp.cpu().numpy())

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        # name_dest_im = './depth1.png'
        # im.save(name_dest_im)

        im_np = np.array(im).transpose(2, 0, 1)  # Convert to (C, H, W)
        im_tensor = torch.from_numpy(im_np).unsqueeze(0)  # Add batch dimension (B, C, H, W)

        # print(im_tensor.shape)  # Should print torch.Size([1, 3, H, W])
        # print("-> depth map is computed ")
        # Scale the tensor to [0, 1]
        im_tensor = im_tensor.float().div(255.0)
        # torchvision.utils.save_image(im_tensor, './depth1.png')
        return im_tensor
        

    # print('-> Done!')


if __name__ == '__main__':
    # args = parse_args()
    
    image_path = "/home/b311/data3/qilishuang/Zero-Shot/datasets/HSTS/depth/0586_rehaze.png"
    model_path = "./MODEL_PATH"
    # Load and preprocess the image
    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size
    input_transform = transforms.Compose([
        # transforms.Resize((256, 256)),  # Resize to the model's expected input size
        transforms.ToTensor()
    ])
    input_tensor = input_transform(input_image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension


    depthmap=inference(input_tensor,model_path)
    torchvision.utils.save_image(depthmap, '/home/b311/data3/qilishuang/Zero-Shot/datasets/HSTS/depth/depth_0586_rehaze.png')
# 
# python inference_single_image.py --image_path="/home/b311/data3/qilishuang/Zero-Shot-RDB/log/rehaze.png" --load_weights_folder=MODEL_PATH