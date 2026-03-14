from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
import numpy as np
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import pandas as pd
from model import Residual,ResNet50
import torch
from torch import nn
import copy
import time
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

class CropDiseaseDataset(Dataset):
    def __init__(self, txt_path, root_dir, transform=None):
        """
        txt_path: 你的 train_list.txt 的完整路径
        root_dir: 图片的根目录（用来和 txt 里的相对路径拼接）
        transform: 图像预处理流水线
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []  # 用来存放 (图片完整路径, 标签) 的名册

        # 1. 打开并解析 txt 文件
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()  # 去除首尾的回车和空格
                if not line:
                    continue

                # 按空格切分，左边是路径，右边是标签
                parts = line.split(' ')
                if len(parts) >= 2:
                    img_rel_path = parts[0]
                    label = int(parts[1])  # 极其重要：必须强转为 int

                    # 抹平 Windows(\) 和 Linux(/) 路径分隔符的差异
                    img_rel_path = img_rel_path.replace('\\', os.sep).replace('/', os.sep)

                    # 拼接绝对路径
                    full_path = os.path.join(self.root_dir, img_rel_path)
                    self.img_info.append((full_path, label))

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        # 2. 按需读取图片和标签
        img_path, label = self.img_info[index]

        # 3. 读取真实图片，并强制转换为 RGB 3通道 (防止灰度图报错)
        img = Image.open(img_path).convert('RGB')

        # 4. 执行预处理操作 (转为 Tensor 等)
        if self.transform:
            img = self.transform(img)

        return img, label

def train_val_data_process():
    TXT_PATH = './data/AgriculturalDisease_trainingset/train_list.txt'  # txt 文件的路径
    ROOT_DIR = './data/'  # 如果txt里的相对路径包含了 "AgriculturalDisease_trainingset"，这里填上一级目录即可

    # 定义图像预处理规则 (适配 ResNet)
    # 这里使用了 ImageNet 标准的均值和方差
    common_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4783, 0.5288, 0.3908], std=[0.2150, 0.2011, 0.2482])
    ])

    # 1. 实例化完整的数据集
    full_dataset = CropDiseaseDataset(txt_path=TXT_PATH, root_dir=ROOT_DIR, transform=common_transform)

    # 2. 划分训练集和验证集 (比如按 8:2 划分)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # 使用 random_split 随机切分，加上 generator 保证每次切分结果一致(固定随机种子42)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 3. 包装成 DataLoader
    # 训练集打乱 (shuffle=True)，验证集不打乱 (shuffle=False)
    train_dataloader = DataLoader(
        dataset=train_dataset,
    batch_size = 32,
    shuffle = True,
    pin_memory = True,
    num_workers = 8,
    persistent_workers = True
    )
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=32,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=8,
                                persistent_workers=True
    )

    print(f"数据加载完成！总图片数: {total_size}。训练集: {train_size} 张，验证集: {val_size} 张。")

    return train_dataloader, val_dataloader



def add_progress_bar(dataloader, epoch, num_epochs, mode="Train"):
    """
    独立的进度条包装函数。
    不破坏原结构，只需将原有的 dataloader 作为参数传入即可。
    """
    desc_text = f"{mode} Epoch {epoch+1}/{num_epochs}"
    # leave=False 表示当前 epoch 跑完后进度条会自动消失，保持控制台整洁
    return tqdm(dataloader, desc=desc_text, leave=False)


def model_train_process(model,train_dataloader,val_dataloader,num_epochs):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    criterion=nn.CrossEntropyLoss()
    model = model.to(device)

    best_acc=0.0
    best_model_wts=copy.deepcopy(model.state_dict())

    train_loss_all=[]
    val_loss_all=[]
    train_acc_all=[]
    val_acc_all=[]

    since = time.time()



    for epoch in range(num_epochs):
        train_loss=0
        val_loss=0
        train_correct=0
        val_correct=0
        train_num=0
        val_num=0



        model.train()
        for step, (b_x, b_y) in enumerate(add_progress_bar(train_dataloader, epoch, num_epochs, mode="Train")):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            output=model(b_x)
            pre_label=torch.argmax(output,dim=1)
            loss=criterion(output,b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss+=loss.item()*b_x.size(0)
            train_correct+=torch.sum(pre_label==b_y.data)
            train_num+=b_x.size(0)

        model.eval()
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(add_progress_bar(val_dataloader, epoch, num_epochs, mode="Val")):
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                output = model(b_x)
                pre_label = torch.argmax(output, dim=1)
                loss = criterion(output, b_y)

                val_loss += loss.item() * b_x.size(0)
                val_correct += torch.sum(pre_label == b_y.data)
                val_num += b_x.size(0)

        train_loss_all.append((train_loss / train_num))
        val_loss_all.append(val_loss / val_num)
        train_acc_all.append(train_correct.double().item() / train_num)
        val_acc_all.append(val_correct.double().item() / val_num)

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print("训练耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, './best_model.pth')

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all
                                       })
    return train_process


def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    save_path = './training_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__=="__main__":
    model=ResNet50(Residual)
    train_dataloader,val_dataloader=train_val_data_process()
    train_process=model_train_process(model,train_dataloader,val_dataloader,50)
    matplot_acc_loss(train_process)





