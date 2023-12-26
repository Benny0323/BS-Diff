import torch.nn.functional as F
from matplotlib import pyplot as plt

import sys
import torch
from diffusers import DDPMScheduler

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import cv2 as cv
import pandas as pd

import time

from dataclasses import dataclass

from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm

from generative.networks.nets import DiffusionModelUNet

from torch_ema import ExponentialMovingAverage

from early_stopping import EarlyStopping


@dataclass
class config():
    batch_size = 2
    learning_rate = 1e-4
    milestones = [25, 40, 70]
    epoch_number = 200
    num_train_timesteps = 1000
    beta_schedule = "squaredcos_cap_v2"
    use_server = False
    in_channels = 2
    out_channels = 2
    image_size = 256
    resume_log_dir = './Aug_Final_ctrain.pth'


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
    file = open('log_Aug_ctrain.txt', 'w')  # 保存日志位置
else:
    file = None  # 取消日志输出

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境

train_file_list = "Aug_train_set.txt"  # 存储训练集文件名的文本文件
validate_file_list = "Aug_validation_set.txt"  # 存储测试集文件名的文本文件

label_path = "./BS_Aug"  # 标签文件夹路径
img_path = "./CXR_Aug"  # 图像文件夹路径
Train_Transform = transforms.Compose(
    [myTransformMethod(),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

Validate_Transform = transforms.Compose(
    [myTransformMethod(),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])
myTrainDataset = myDataset(train_file_list, label_path, img_path, Train_Transform)  # 创建训练集实例
myValidateDataset = myDataset(validate_file_list, label_path, img_path, Validate_Transform)  # 创建测试集实例

myTrainDataLoader = DataLoader(myTrainDataset, batch_size=config.batch_size, shuffle=True)  # 创建数据加载器实例
myValidateDataLoader = DataLoader(myValidateDataset, batch_size=config.batch_size, shuffle=True)  # 创建数据加载器实例

print("batch数量:", len(myTrainDataLoader))  # 输出batch数量
print("数据集大小:", len(myTrainDataset))  # 输出数据集大小

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

model = model.to(device)
print("模型参数量:", sum([p.numel() for p in model.parameters()]))

# 设定噪声调度器
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0015, beta_end=0.0205, beta_schedule=config.beta_schedule)
# 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
config.milestones = [x * len(myTrainDataLoader) for x in config.milestones]
# print(config.milestones)
optimizerScheduler = MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)  # 动态设置学习率

ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

train_losses = []
validate_losses = []

plt_train_loss_epoch = []
plt_validate_loss_epoch = []

train_epoch_list = list(range(0, config.epoch_number))

validation_epoch_list = list(range(0, int(config.epoch_number / 10)))

if __name__ == "__main__":
    # 训练过程
    print(time.strftime("%H:%M:%S", time.localtime()), "----------Begin Training----------", file=file)


    for epoch in range(config.epoch_number):
        model.train()
        print(time.strftime("%H:%M:%S", time.localtime()),
              f"Epoch:{epoch},learning rate:{optimizer.param_groups[0]['lr']}",
              file=file)
        for i, batch in tqdm(enumerate(myTrainDataLoader)):
            images, labels = batch[0].to(device), batch[1].to(device)
            blank = torch.randn_like(images).to(device)
            cat = torch.cat((labels, images), dim=-3)
            # 为图片添加噪声
            noise = torch.randn(images.shape).to(images.device)
            noise = torch.cat((noise, blank), dim=-3)
            bs = images.shape[0]

            # 为每张图片随机采样一个时间步
            timesteps = torch.randint(0, config.num_train_timesteps, (bs,), device=images.device).long()

            # 根据每个时间步的噪声幅度，向清晰的图片中添加噪声
            noisy_images = noise_scheduler.add_noise(cat, noise, timesteps)

            # 获取模型的预测结果
            noise_pred = model(noisy_images, timesteps)

            # 计算损失
            if noise_scheduler.prediction_type == "v_prediction":
            # Use v-prediction parameterization
                target = noise_scheduler.get_velocity(cat, noise, timesteps)
            elif noise_scheduler.prediction_type == "epsilon":
                target = noise

            train_loss = F.mse_loss(noise_pred[:, 0].float(), target[:, 0].float())
            train_loss.backward(train_loss)
            train_losses.append(train_loss.item())

            # 迭代模型参数, 更新完参数后，同步update shadow weights
            optimizer.step()
            optimizer.zero_grad()
            ema.update()
            # 更新动态学习率


        if (epoch + 1) % 1 == 0:
            train_loss_epoch = sum(train_losses[-len(myTrainDataLoader):]) / len(myTrainDataLoader)
            print(time.strftime("%H:%M:%S", time.localtime()),
                  f"Epoch:{epoch},train losses:{train_loss_epoch}",
                  file=file)
            plt_train_loss_epoch.append(train_loss_epoch)

        if epoch % 10 == 0 :
            model.eval()
            with ema.average_parameters():
                # validate_epoch.append(epoch)
                print(time.strftime("%H:%M:%S", time.localtime()), "----------Stop Training----------", file=file)
                print(time.strftime("%H:%M:%S", time.localtime()), "----------Begin Validation----------",
                      file=file)
                with torch.no_grad():
                    for i, batch in enumerate(myValidateDataLoader):
                        if (i + 1) % 1 == 0:
                            print(time.strftime("%H:%M:%S", time.localtime()), "Validate Batch", i)
                        images, labels = batch[0].to(device), batch[1].to(device)
                        blank = torch.randn_like(images).to(device)
                        cat = torch.cat((labels, images), dim=-3)
                        # 为图片添加噪声
                        noise = torch.randn(images.shape).to(images.device)
                        noise = torch.cat((noise, blank), dim=-3)
                        bs = images.shape[0]

                        # 为每张图片随机采样一个时间步
                        timesteps = torch.randint(0, config.num_train_timesteps, (bs,), device=images.device).long()

                        # 根据每个时间步的噪声幅度，向清晰的图片中添加噪声
                        noisy_images = noise_scheduler.add_noise(cat, noise, timesteps)

                        # 获取模型的预测结果
                        noise_pred = model(noisy_images, timesteps)

                        # 计算损失
                        if noise_scheduler.prediction_type == "v_prediction":
                            # Use v-prediction parameterization
                            target = noise_scheduler.get_velocity(cat, noise, timesteps)
                        elif noise_scheduler.prediction_type == "epsilon":
                            target = noise

                        validate_loss = F.mse_loss(noise_pred[:, 0].float(), target[:, 0].float())
                        validate_losses.append(validate_loss.item())  # 取一个张量里面的元素的具体值

                validate_loss_epoch = sum(validate_losses[-len(myValidateDataLoader):]) / len(myValidateDataLoader)
                print(time.strftime("%H:%M:%S", time.localtime()),
                      f"Epoch:{epoch},test losses:{validate_loss_epoch}")
                plt_validate_loss_epoch.append(validate_loss_epoch)

                optimizerScheduler.step()

                print(time.strftime("%H:%M:%S", time.localtime()), "----------End Validation----------", file=file)
                early_stopping = EarlyStopping(config.resume_log_dir, epoch, optimizer, ema)
                early_stopping(validate_loss_epoch, model)
                # 达到早停止条件时，early_stop会被置为True
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch{epoch}")
                    sys.exit()  # 跳出迭代，结束训练
                print(time.strftime("%H:%M:%S", time.localtime()), "----------Continue to Train----------",
                      file=file)
    print(time.strftime("%H:%M:%S", time.localtime()), "----------End Training----------", file=file)

    # 查看损失曲线
    f, ([ax1, ax2]) = plt.subplots(1, 2, sharex=True)
    ax1.plot(train_epoch_list, plt_train_loss_epoch, color="red", label="train_loss")
    ax1.set_title('Train loss during training')
    ax2.plot(validation_epoch_list, plt_validate_loss_epoch, color="blue", label="validate_loss")
    ax2.set_title('Validation loss during training')
    plt.legend()
    # plt.grid()
    if not config.use_server:
        plt.savefig("./Results/Aug_Final_ctrain_loss.png")  # 保存损失曲线
    else:
        plt.show()  # 展示损失曲线
