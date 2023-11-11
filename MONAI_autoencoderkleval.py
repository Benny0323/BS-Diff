import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import torch

from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.networks.layers import Act
from monai.utils import first, set_determinism
from torch.nn import L1Loss
from tqdm import tqdm
import pandas as pd
import cv2 as cv

from torchvision import transforms
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

print_config()

# for reproducibility purposes set a seed
set_determinism(42)


class config():
    use_server = False
    in_channels = 1
    out_channels = 1
    image_size = 256

class myTransform():  # Python3默认继承object类
    def __call__(self, img):  # __call___，让类实例变成一个可以被调用的对象，像函数
        img = cv.resize(img, (config.image_size, config.image_size))  # 改变图像大小
        if img.shape[-1] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将BGR(openCV默认读取为BGR)改为GRAY
        return img  # 返回预处理后的图像


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境


Test_Transform = transforms.Compose(
    [myTransform(),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])



model = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256, 384, 512),
    latent_channels=8,
    num_res_blocks=2,
    norm_num_groups=32,
    attention_levels=(False, True, True, True),
)
model.to(device)


optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)


if __name__ == "__main__":
    model = torch.load("MONAI_autoencoderkl.pth").to(device)
    model.eval()

    with (torch.no_grad()):

        img_path = "./MONAI_save_bs"  # 测试集图像路径
        save_path = "./Final"

        for filename in os.listdir(img_path):
            cxr_path = img_path + "/" + filename
            input = cv.imread(cxr_path)  # 512 * 512 * 3
            input = Test_Transform(input).to(device)  # 256 * 256
            input = torch.unsqueeze(input, dim=-3)

            reconstruction, _, _ = model(input)
            reconstruction = torch.squeeze(reconstruction, dim=-3)
            reconstruction = np.array(reconstruction.detach().to("cpu"))
            reconstruction = np.transpose(reconstruction, (1, 2, 0))  # C*H*W -> H*W*C
            reconstruction = reconstruction * 0.5 + 0.5
            reconstruction = np.clip(reconstruction, 0, 1)
            reconstruction *= 255
            reconstruction = cv.resize(reconstruction, (512, 512))
            save_bs_path = save_path + "/" + filename
            cv.imwrite(save_bs_path, reconstruction)

