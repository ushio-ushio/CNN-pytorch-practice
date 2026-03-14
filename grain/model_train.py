from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
import numpy as np
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import pandas as pd
from model import Residual,ResNet_18
import torch
from torch import nn
import copy
import time
from tqdm import tqdm

def train_val_data_process():
    ROOT_TRAIN=r'data\train'

    normalize=transforms.Normalize([0.042,0.0428,0.0441],[0.033,0.0343, 0.0363])

    train_transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])

    train_data=ImageFolder(ROOT_TRAIN,transform=train_transform)

    train_data,val_data=Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])

    train_dataloader=Data.DataLoader(dataset=train_data,
                                     batch_size=32,
                                     shuffle=True,
                                     pin_memory=True,
                                     num_workers=8,
                                     persistent_workers=True)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=32,
                                       shuffle=True,
                                     pin_memory=True,
                                     num_workers=8,
                                     persistent_workers=True)

    return train_dataloader,val_dataloader

def add_progress_bar(dataloader, epoch, num_epochs, mode="Train"):
    """
    独立的进度条包装函数。
    不破坏原结构，只需将原有的 dataloader 作为参数传入即可。
    """
    desc_text = f"{mode} Epoch {epoch}/{num_epochs-1}"
    # leave=False 表示当前 epoch 跑完后进度条会自动消失，保持控制台整洁
    return tqdm(dataloader, desc=desc_text, leave=False)

def train_model_process(model,train_dataloader,val_dataloader,num_epochs):

    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

    criterion=nn.CrossEntropyLoss()

    model=model.to(device)

    best_model_wts=copy.deepcopy(model.state_dict())

    best_acc=0.0
    train_loss_all=[]
    val_loss_all=[]
    train_acc_all = []
    val_acc_all = []

    since=time.time()
    for epoch in range(num_epochs):
        print("epoch {}/{}".format(epoch,num_epochs-1))
        print("-"*10)
        train_loss=0.0
        train_corrects=0
        val_loss = 0.0
        val_corrects = 0
        train_num=0
        val_num=0

        model.train()
        for step,(b_x,b_y) in enumerate(add_progress_bar(train_dataloader, epoch, num_epochs, mode="Train")):
            b_x=b_x.to(device)
            b_y=b_y.to(device)

            output=model(b_x)

            pre_lab=torch.argmax(output,dim=1)

            loss=criterion(output,b_y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss+=loss.item()*b_x.size(0)
            train_corrects+=torch.sum(pre_lab==b_y.data)
            train_num+=b_x.size(0)

        model.eval()
        with torch.no_grad():
         for step,(b_x,b_y) in enumerate(add_progress_bar(val_dataloader, epoch, num_epochs, mode="Val")):
            b_x=b_x.to(device)
            b_y=b_y.to(device)


            output=model(b_x)
            pre_lab=torch.argmax(output,dim=1)
            loss=criterion(output,b_y)

            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)

        train_loss_all.append((train_loss/train_num))
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_acc_all.append(val_corrects.double().item()/val_num)

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())

        time_use=time.time()-since
        print("训练耗费的时间{:.0f}m{:.0f}s".format(time_use//60,time_use%60))

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, './best_model.pth')

    train_process=pd.DataFrame(data={"epoch":range(num_epochs),
                                         "train_loss_all":train_loss_all,
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
    model=ResNet_18(Residual)
    train_dataloader,val_dataloader=train_val_data_process()
    train_process=train_model_process(model,train_dataloader,val_dataloader,50)
    matplot_acc_loss(train_process)
