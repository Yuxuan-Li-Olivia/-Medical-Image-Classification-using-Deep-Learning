import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 加载预训练的ResNet18模型
model = models.resnet18(pretrained=True)
model.eval()  # 设置为评估模式

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.Grayscale(num_output_channels=3),  # 将黑白图像转换为3通道
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载黑白图片
image_path = '/mnt/e/COVID19_Classification/covid19_rawdata/COVID-19+Aug+CGAN/test/COVID/2020.03.22.20040782-p25-1544.png'
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度

# 定义 hook 函数，用于提取特征图
feature_maps = []

def hook_fn(module, input, output):
    feature_maps.append(output)

# 注册 hook 到 ResNet18 的某一层（例如 layer1）
layer = model.layer1
hook = layer.register_forward_hook(hook_fn)

# 前向传播，提取特征图
with torch.no_grad():
    model(input_tensor)

# 取消 hook
hook.remove()

# 可视化特征图
plt.figure(figsize=(10, 10))
for i in range(feature_maps[0].shape[1]):  # 遍历每个通道
    plt.subplot(8, 8, i + 1)  # 假设特征图有 64 个通道，8x8 排列
    plt.imshow(feature_maps[0][0, i, :, :], cmap='viridis')
    plt.axis('off')
plt.show()