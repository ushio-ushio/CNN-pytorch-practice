import torch
from torch import nn
from torchsummary import summary



class Residual(nn.Module):
    def __init__(self,in_channels,num_channels,stride=1,use_conv=False):
        super(Residual, self).__init__()
        self.ReLu=nn.ReLU()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=num_channels//4,padding=0,kernel_size=1)
        self.conv2=nn.Conv2d(in_channels=num_channels//4,out_channels=num_channels//4,padding=1,stride=stride,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=num_channels//4, out_channels=num_channels, padding=0, kernel_size=1)
        self.bn1=nn.BatchNorm2d(num_channels//4)
        self.bn2 = nn.BatchNorm2d(num_channels//4)
        self.bn3 = nn.BatchNorm2d(num_channels)
        if use_conv:
            self.conv4=nn.Conv2d(in_channels=in_channels,out_channels=num_channels,padding=0,stride=stride,kernel_size=1)
            self.bn4=nn.BatchNorm2d(num_channels)
        else:
            self.conv4 =None
            self.bn4 = None

    def forward(self,x):
        y=self.ReLu(self.bn1(self.conv1(x)))
        y=self.ReLu(self.bn2(self.conv2(y)))
        y=self.bn3(self.conv3(y))
        if self.conv4:
            x=self.bn4(self.conv4(x))
        y=self.ReLu( y+x)
        return y




class ResNet50(nn.Module):
    def __init__(self,Residual):
        super(ResNet50,self).__init__()
        self.b1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        )
        self.b2=nn.Sequential(
            Residual(64,256,stride=1,use_conv=True),
            Residual(256, 256, stride=1, use_conv=False),
            Residual(256, 256, stride=1, use_conv=False)
        )
        self.b3=nn.Sequential(
            Residual(256, 512, stride=2, use_conv=True),
            Residual(512, 512, stride=1, use_conv=False),
            Residual(512, 512, stride=1, use_conv=False),
            Residual(512, 512, stride=1, use_conv=False)
        )
        self.b4=nn.Sequential(
            Residual(512, 1024, stride=2, use_conv=True),
            Residual(1024, 1024, stride=1, use_conv=False),
            Residual(1024, 1024, stride=1, use_conv=False),
            Residual(1024, 1024, stride=1, use_conv=False),
            Residual(1024, 1024, stride=1, use_conv=False),
            Residual(1024, 1024, stride=1, use_conv=False)
        )
        self.b5=nn.Sequential(
            Residual(1024, 2048, stride=2, use_conv=True),
            Residual(2048, 2048, stride=1, use_conv=False),
            Residual(2048, 2048, stride=1, use_conv=False)
        )
        self.b6=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048,61)
        )
    def forward(self,x):
        x=self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet50(Residual).to(device)
    print(summary(model, (3, 224, 224)))
