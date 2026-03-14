# CNN-pytorch-practice
关于我学习pytorch框架针对CNN内容的实战项目，还有数维杯Resnet识别图片的实战项目


## 1. 项目简介

本项目旨在复现经典的卷积神经网络（CNN）模型，并将其应用于真实的计算机视觉任务中。仓库内不仅包含了多种经典图像分类网络的 PyTorch 实现，还重点记录了基于 ResNet50 在“数维杯”农作物病害分类（Agricultural Disease Classification）项目中的完整实战过程。


## 2. 包含的模型实现

本仓库从零构建并测试了以下经典 CNN 架构：

* **AlexNet**: 探索了早期深度卷积网络的基本结构与 ReLU、Dropout 等机制。
* **VGG (Visual Geometry Group)**: 实现了基于堆叠小卷积核（3x3）的深层网络结构。
* **GoogLeNet (Inception v1)**: 构建了 Inception 模块，实践了多尺度特征提取与辅助分类器。
* **ResNet18 & ResNet50**: 实现了基于残差块（Residual Block）和瓶颈结构（Bottleneck）的深度残差网络，有效解决了深层网络的梯度消失问题。

## 3. 核心实战：数维杯农作物病害分类

本项目的主要应用落地是针对“数维杯”提供的大规模农作物病害数据集进行分类。该任务要求模型能够准确识别 61 种不同农作物及其病害状态。实战采用了 **ResNet50** 作为主干网络进行训练与推断。

### 3.1 数据处理流水线

* **自定义 Dataset**: 编写了 `CropDiseaseDataset` 类，支持从 txt 标签文件中动态读取相对路径并拼接绝对路径，适配 Windows/Linux 多平台。
* **数据预处理 (Transforms)**:
* 图像统一 Resize 至 224x224，满足 ResNet50 的输入尺度要求。
* 基于训练集数据分布，独立计算并应用了特定的均值（Mean）和标准差（Std）进行 Normalize 归一化处理，而非盲目套用 ImageNet 的默认参数。


* **DataLoader 优化**: 在训练环节开启了 `pin_memory=True`（锁页内存）、`num_workers` 多进程数据加载以及 `persistent_workers=True`，大幅降低了 CPU 与 GPU 之间的数据传输瓶颈，使单个 Epoch 的训练时间保持高度稳定。

### 3.2 模型训练与评估

* **环境与超参数**: 基于 PyTorch 框架，使用 Adam 优化器，损失函数采用 CrossEntropyLoss。
* **训练策略**:
* 采用 `model.train()` 与 `model.eval()` 严格控制 BatchNorm 和 Dropout 层在不同阶段的行为。
* 实现了基于 `torch.no_grad()` 的验证集零梯度推断，有效节省显存。
* 包含最佳权重自动保存逻辑（保存验证集准确率最高的 `best_model.pth`）。


* **当前基线性能**:
* 在 50 个 Epoch 的训练后，训练集准确率可达 99% 以上。
* 验证集/测试集基线准确率稳定在 **83% - 84%** 左右。


* **后续优化方向**: 当前基线模型在训练后期表现出一定的过拟合倾向。后续计划引入数据增强（Data Augmentation，如随机裁剪、翻转等）、学习率衰减策略（LR Scheduler）以及权重衰减（Weight Decay）以进一步提升泛化能力，冲击 90% 以上的测试准确率。
