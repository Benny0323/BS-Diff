import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
import torch
from torch import nn
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel, PNDMScheduler
from torchvision import transforms

import os
import cv2 as cv
import pandas as pd

import time

from dataclasses import dataclass

from tqdm import tqdm

from generative.networks.nets import DiffusionModelUNet
from generative.inferers import DiffusionInferer

@dataclass
class config():
    num_train_timesteps = 1000
    beta_schedule = "squaredcos_cap_v2"
    in_channels = 2
    out_channels = 2
    image_size = 256


class myTransformMethod():  # Python3默认继承object类
    def __call__(self, img):  # __call___，让类实例变成一个可以被调用的对象，像函数
        img = cv.resize(img, (config.image_size, config.image_size))  # 改变图像大小
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将BGR(openCV默认读取为BGR)改为GRAY
        return img  # 返回预处理后的图像

# UNet2DModel输入图像和timestep,输出图像

model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=2,
    out_channels=2,
    num_res_blocks=3,
    num_channels=[64, 128, 256, 512, 768],
    attention_levels=[False, True, True, True, True],
    num_head_channels=[0, 128, 256, 512, 768],
    transformer_num_layers=3,
)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0015, beta_end=0.0205, beta_schedule=config.beta_schedule)
# 训练循环
noise_scheduler.set_timesteps(num_inference_steps=1000)

myTransform = transforms.Compose([myTransformMethod(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    m_dict = torch.load('./Aug_Final_ctrain.pth')
    model.load_state_dict(m_dict['model'])
    model.eval()
    model.to(device)

    img_path = "./Test_CXR"  # 测试集图像路径
    save_path = "./Aug_save_bs"

    for filename in os.listdir(img_path):
        cxr_path = img_path + "/" + filename
        CXR = cv.imread(cxr_path)
        CXR = cv.resize(CXR, (config.image_size, config.image_size))  # 改变图像大小
        CXR = myTransform(CXR).to(device)
        CXR_T = CXR.shape[0]

        noise = torch.randn(CXR.shape).to(CXR.device) # 1 * H * W
        sample = torch.cat((noise, CXR), dim=-3) # 2 * H * W
        sample = torch.unsqueeze(sample, dim=-4) # 1 * 2 * H * W

        for j, t in tqdm(enumerate(noise_scheduler.timesteps)):
            sample = noise_scheduler.scale_model_input(sample, t)
            with torch.no_grad():
                #cxr_time = torch.Tensor((t,)).to(sample.device).long().repeat(CXR_T)
                #noisy_images = noise_scheduler.add_noise(sample[:, 1], noise, cxr_time)
                #sample[:, 1] = noisy_images
                residual = model(sample, timesteps=torch.Tensor((t,)).to(sample.device).long())  # predict the noise
            sample = noise_scheduler.step(residual, t, sample).prev_sample # 1 * 2 * H * W
            sample = torch.squeeze(sample, dim=-4)  # 2 * H * W
            sample = sample[0]  # sample的预测x_0结果 H * W
            sample = torch.unsqueeze(sample, dim=-3)  # 1 * H * W
            sample = torch.cat((sample, CXR), dim=-3)  # 2 * H * W
            sample = torch.unsqueeze(sample, dim=-4)  # 1 * 2 * H * W

        clean = torch.squeeze(sample, dim=-4)
        clean = clean[0]
        clean = torch.unsqueeze(clean, dim=-3)
        clean = np.array(clean.detach().to("cpu"))
        clean = np.transpose(clean, (1, 2, 0))  # C*H*W -> H*W*C
        clean = clean * 0.5 + 0.5
        clean *= 255
        save_bs_path = save_path + "/" + filename
        cv.imwrite(save_bs_path, clean)
