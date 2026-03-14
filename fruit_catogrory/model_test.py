import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import GoogLeNet,Inception
from  torchvision.datasets import ImageFolder
from PIL import Image


def test_data_process():
    ROOT_TRAIN=r'data\test'
    normalize = transforms.Normalize([0.229, 0.196,  0.143],[0.099,0.0800, 0.0660])

    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=16,
                                       shuffle=False,
                                       num_workers=0)
    return test_dataloader


def test_model_process(model, test_dataloader):
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # 讲模型放入到训练设备中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output= model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确度test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)

if __name__=="__main__":
    model=GoogLeNet(Inception)
    model.load_state_dict(torch.load('best_model.pth',weights_only=True))
    # 加载测试数据
    # test_dataloader = test_data_process()
    # # 加载模型测试的函数
    # test_model_process(model, test_dataloader)

    #
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    image=Image.open('grren.png').convert('RGB')
    normalize = transforms.Normalize([0.229, 0.196,  0.143],[0.099,0.0800, 0.0660])

    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    image=test_transform(image)
    image=image.unsqueeze(0)

    classes = ['apple', 'banana','grape','orange','pear']

    with torch.no_grad():
        model.eval()
        image=image.to(device)
        output=model(image)
        pre_lab=torch.argmax(output,dim=1)
        result=pre_lab.item()
    print("预测值：",  classes[result])
    #








    # classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # with torch.no_grad():
    #     for b_x, b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #
    #         # 设置模型为验证模型
    #         model.eval()
    #         output = model(b_x)
    #         pre_lab = torch.argmax(output, dim=1)
    #         result = pre_lab.item()
    #         label = b_y.item()
    #         print("预测值：",  classes[result], "------", "真实值：", classes[label])