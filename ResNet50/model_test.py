import torch
import torch.utils.data as Data
from accelerate.test_utils.scripts.test_distributed_data_loop import test_data_loader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import Residual,ResNet50
from  torchvision.datasets import ImageFolder
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader, random_split

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

def test_data_process():

        # 1. 请根据实际情况确认这两个路径
        TXT_PATH = './data/AgriculturalDisease_validationset/ttest_list.txt'
        # 注意这里：txt里已经自带了 'AgriculturalDisease_validationset/images'
        # 所以 ROOT_DIR 依然填上一级目录即可，拼起来刚好对上！
        ROOT_DIR = './data/'

        # 2. 定义预处理规则
        # 核心原则：测试集的 Resize 和 Normalize 必须和训练集【完全一致】！
        # 如果你刚才用脚本算出了自己的 mean 和 std，记得把这里替换掉
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4783, 0.5288, 0.3908], std=[0.2150, 0.2011, 0.2482])
        ])

        # 3. 直接复用我们写好的类，实例化测试集
        test_dataset = CropDiseaseDataset(txt_path=TXT_PATH, root_dir=ROOT_DIR, transform=test_transform)

        # 4. 包装成 DataLoader
        # 【致命细节】：测试集的 shuffle 必须设置为 False！我们不需要打乱试卷。
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0)

        print(f"✅ 测试集加载完成！总共待测图片数: {len(test_dataset)} 张。")

        return test_dataloader



def test_model_process(model,test_dataloader):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)

    test_correct=0
    test_num=0

    model.eval()
    with torch.no_grad():
        for step,(test_data_x,test_data_y) in enumerate(test_dataloader):
            test_data_x=test_data_x.to(device)
            test_data_y=test_data_y.to(device)
            output=model(test_data_x)
            pre_label=torch.argmax(output,dim=1)

            test_correct+=torch.sum(pre_label==test_data_y.data)
            test_num+=test_data_x.size(0)

    test_acc=test_correct.double().item()/test_num
    print("模型准确率为:",test_acc)

if __name__=="__main__":
    model = ResNet50(Residual)
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    # 加载测试数据
    test_dataloader = test_data_process()
    # 加载模型测试的函数
    test_model_process(model, test_dataloader)

