from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
import numpy as np
train_data =FashionMNIST(root='./data'
                         ,train=True,
                         transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                         download=True)

train_loader =Data.DataLoader(dataset=train_data,
                              batch_size=64,
                              shuffle=True,
                              num_workers=0)

for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
batch_y = b_y.numpy()  # 将张量转换成Numpy数组
class_label = train_data.classes  # 训练集的标签
print(class_label)
# print("The size of batch in train data:", batch_x.shape)  # 每个mini-batch的维度是64*224*224