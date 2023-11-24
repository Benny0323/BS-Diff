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
import pytorch_msssim

from torchvision import transforms
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

print_config()

# for reproducibility purposes set a seed
set_determinism(42)


class config():
    batch_size = 2
    epoch_number = 200
    use_server = False
    in_channels = 1
    out_channels = 1
    image_size = 256


class myDataset(Dataset):  # 定义数据集类
    def __init__(self, filelist, label_dir, img_dir, transform=None):  # 传入参数(标签路径,图像路径,图像预处理方式,标签预处理方式)
        self.label_dir = label_dir  # 读取标签路径
        self.img_dir = img_dir  # 读取图像路径
        self.transform = transform  # 读取图像预处理方式
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)  # 读取文件名列表

    def __len__(self):
        return len(self.filelist)  # 读取文件名数量作为数据集长度

    def __getitem__(self, idx):  # 从数据集中取出数据
        label_path = self.label_dir  # 读取标签文件夹路径
        img_path = self.img_dir  # 读取图片文件夹路径

        file = self.filelist.iloc[idx, 0]  # 读取文件名
        # print(file)
        image = cv.imread(os.path.join(img_path, file))  # 用openCV的imread函数读取图像
        label = cv.imread(os.path.join(label_path, file))  # 用openCV的imread函数读取标签

        if self.transform:
            image = self.transform(image)  # 图像预处理
            label = self.transform(label)  # 标签预处理
        return image, label  # 返回图像和标签


class myTransformMethod():  # Python3默认继承object类
    def __call__(self, img):  # __call___，让类实例变成一个可以被调用的对象，像函数
        img = cv.resize(img, (config.image_size, config.image_size))  # 改变图像大小
        if img.shape[-1] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将BGR(openCV默认读取为BGR)改为GRAY
        return img  # 返回预处理后的图像


if config.use_server:
    file = open('Aug_MONAI_AutoencoderKL_log.txt', 'w')  # 保存日志位置
else:
    file = None  # 取消日志输出

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境

train_file_list = "train_set_new.txt"  # 存储训练集文件名的文本文件
validate_file_list = "test_set_new.txt"  # 存储测试集文件名的文本文件

img_path = "./Aug_MONAI_save_bs"  # 图像文件夹路径
label_path = "./New_BS"  # 标签文件夹路径

myTransform = transforms.Compose(
    [myTransformMethod(),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])
myTrainDataset = myDataset(train_file_list, label_path, img_path, myTransform)  # 创建训练集实例
myTestDataset = myDataset(validate_file_list, label_path, img_path, myTransform)  # 创建测试集实例

myTrainDataLoader = DataLoader(myTrainDataset, batch_size=config.batch_size, shuffle=True)  # 创建数据加载器实例
myValidateDataLoader = DataLoader(myTestDataset, batch_size=config.batch_size, shuffle=True)  # 创建数据加载器实例

print("batch数量:", len(myTrainDataLoader))  # 输出batch数量
print("数据集大小:", len(myTrainDataset))  # 输出数据集大小

# ## Define the network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
model = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256, 384),
    latent_channels=8,
    num_res_blocks=2,
    norm_num_groups=32,
    attention_levels=(False, True, True),
)
model.to(device)

discriminator = PatchDiscriminator(
    spatial_dims=2,
    num_layers_d=3,
    num_channels=64,
    in_channels=1,
    out_channels=1,
    kernel_size=4,
    activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
    norm="BATCH",
    bias=False,
    padding=1,
)
discriminator.to(device)

perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="vgg")
perceptual_loss.to(device)

optimizer_g = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=5e-4)

l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")
adv_weight = 0.01
perceptual_weight = 0.001
ms_ssim_weight = 1.0

# ## Model Training

#
n_epochs = config.epoch_number
val_interval = 25
epoch_recon_loss_list = []
epoch_gen_loss_list = []
epoch_disc_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

total_start = time.time()
for epoch in range(n_epochs):
    model.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    progress_bar = tqdm(enumerate(myTrainDataLoader), total=len(myTrainDataLoader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images, labels = batch[0].to(device), batch[1].to(device)
        optimizer_g.zero_grad(set_to_none=True)

        reconstruction, z_mu, z_sigma = model(images)

        recons_loss = l1_loss(reconstruction.float(), labels.float())

        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
        p_loss = perceptual_loss(reconstruction.float(), labels.float())
        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        msssim = pytorch_msssim.MSSSIM(window_size=11, size_average=True, channel=1, normalize='relu')
        msssim_loss = 1 - msssim(reconstruction.float(), labels.float())
        loss_g = recons_loss + perceptual_weight * p_loss + adv_weight * generator_loss + ms_ssim_weight * msssim_loss

        loss_g.backward()
        optimizer_g.step()

        # Discriminator part
        optimizer_d.zero_grad(set_to_none=True)

        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = discriminator(labels.contiguous().detach())[-1]
        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = adv_weight * discriminator_loss

        loss_d.backward()
        optimizer_d.step()

        epoch_loss += recons_loss.item()
        gen_epoch_loss += generator_loss.item()
        disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )
    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
    epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(myValidateDataLoader, start=1):
                images, labels = batch[0].to(device), batch[1].to(device)
                reconstruction, _, _ = model(images)

                # get the first sammple from the first validation batch for visualisation
                # purposes
                if val_step == 1:
                    intermediary_images.append(reconstruction[:n_example_images, 0])

                recons_loss = l1_loss(reconstruction.float(), labels.float())

                val_loss += recons_loss.item()

        val_loss /= val_step
        val_recon_epoch_loss_list.append(val_loss)
        torch.save(model, "Aug_MONAI_autoencoderkl.pth")

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")
# -

# ## Evaluate the training
# ### Visualise the loss

plt.style.use("seaborn-v0_8")
plt.title("Aug_MONAI_Learning Curves", fontsize=20)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_recon_loss_list, color="C0", linewidth=2.0, label="Train")
plt.plot(
    np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
    val_recon_epoch_loss_list,
    color="C1",
    linewidth=2.0,
    label="Validation",
)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig("Aug_MONAI_Learning Curves.png")

# %%
plt.title("Aug_MONAI_Adversarial Training Curves", fontsize=20)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig("Aug_MONAI_Adversarial Training Curves.png")

# ### Visualise some reconstruction images

# # Plot every evaluation as a new line and example as columns
# val_samples = np.linspace(val_interval, n_epochs, int(n_epochs / val_interval))
# fig, ax = plt.subplots(nrows=len(val_samples), ncols=1, sharey=True)
# for image_n in range(len(val_samples)):
#     reconstructions = torch.reshape(intermediary_images[image_n], (64 * n_example_images, 64)).T
#     ax[image_n].imshow(reconstructions.cpu(), cmap="gray")
#     ax[image_n].set_xticks([])
#     ax[image_n].set_yticks([])
#     ax[image_n].set_ylabel(f"Epoch {val_samples[image_n]:.0f}")


# %%
fig, ax = plt.subplots(nrows=1, ncols=2)
images = (images[0, 0] * 0.5 + 0.5) * 255
ax[0].imshow(images.detach().cpu(), vmin=0, vmax=255, cmap="gray")
ax[0].axis("off")
ax[0].title.set_text("Inputted Image")
reconstructions = (reconstruction[0, 0] * 0.5 + 0.5) * 255
ax[1].imshow(reconstructions.detach().cpu(), vmin=0, vmax=255, cmap="gray")
ax[1].axis("off")
ax[1].title.set_text("Reconstruction")
plt.savefig("Aug_MONAI_reconstruction images.png")


