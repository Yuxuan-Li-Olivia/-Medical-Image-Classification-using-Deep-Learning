import torch
from torchvision import datasets, transforms

def load_data(data_dir, image_size=256, batch_size=8):
    # 计算均值和标准差
    temp_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),# 将图像调整为统一的大小,本身就是黑白图片，无需转化为灰度图片
        transforms.ToTensor(), # 将图像转换为PyTorch张量 
    ])
    temp_dataset = datasets.ImageFolder(root=f"{data_dir}/train+Aug+CGAN", transform=temp_transform)
    #它是一个可迭代的对象，每次迭代会返回一个包含图像数据和标签的元组 (images, labels)。images 是一个形状为 [channels, height, width] 的张量
    # （对于灰度图，channels=1），labels 是一个整数，表示图像所属的类别。
    mean, std = 0, 0
    for image, _ in temp_dataset:
        mean += image.mean()
        std += image.std()
    mean /= len(temp_dataset)
    std /= len(temp_dataset)

    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),#统一大小
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),#转换为pytorch张量
        transforms.Normalize(mean=[mean.item()], std=[std.item()]),
    ])

    # 加载数据集
    # 数据集文件夹已经按训练集、验证集和测试集划分好了，可以直接分别加载
    # datasets.ImageFolder 会自动将指定 root 目录下的子文件夹名称作为分类标签
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train+Aug+CGAN", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val+Aug+CGAN", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    # 创建数据加载器
    batch_size=8
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4)

    return train_loader, val_loader, test_loader