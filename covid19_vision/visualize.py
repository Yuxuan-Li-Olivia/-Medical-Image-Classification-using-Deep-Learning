import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from matplotlib.font_manager import FontProperties
 
# 设置我们需要用到的中文字体（字体文件地址）
my_font = FontProperties(fname=r"/mnt/c/windows/fonts/SimHei.ttf", size=12)

#绘制训练曲线（损失和准确率曲线）
def plot_training_curves(train_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='training loss')
    plt.xlabel('Epoch',)
    plt.ylabel('损失',fontproperties=my_font)
    plt.title('训练损失曲线',fontproperties=my_font)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='training acc')
    plt.plot(val_accs, label='validation acc')
    plt.xlabel('Epoch')
    plt.ylabel('准确率',fontproperties=my_font)
    plt.title('准确率曲线',fontproperties=my_font)
    plt.legend()
    plt.show()
#混淆矩阵，用于评估模型性能
def plot_confusion_matrix(net, test_loader, device):
    net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_hat = net(X).argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_hat.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('预测标签',fontproperties=my_font)
    plt.ylabel('真实标签',fontproperties=my_font)
    plt.title('混淆矩阵',fontproperties=my_font)
    plt.show()
#展示样本预测结果
def show_sample_predictions(net, test_loader, device, num_samples=5):
    net.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_hat = net(X).argmax(1)
            X = X.cpu()
            y = y.cpu()
            y_hat = y_hat.cpu()
            break
    plt.figure(figsize=(15, 3))
    for i in range(min(num_samples, X.shape[0])):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[i].permute(1, 2, 0).squeeze(), cmap='gray')
        plt.title(f'真实: {y[i]}\n预测: {y_hat[i]}',fontproperties=my_font)
        plt.axis('off')
    plt.show()
#特征图可视化，观察卷积层提取的特征，理解模型的学习过程
def visualize_feature_maps(net, test_loader, device, layer_index=0):
    net.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            features = net[:layer_index + 1](X)
            break

    plt.figure(figsize=(12, 8))
    for i in range(min(16, features.shape[1])):
        plt.subplot(4, 4, i + 1)
        plt.imshow(features[0, i].cpu(), cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'第 {layer_index} 层特征图',fontproperties=my_font)
    plt.show()