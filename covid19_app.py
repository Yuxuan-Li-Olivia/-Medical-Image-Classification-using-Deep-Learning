import argparse
import torch
from covid19_vision import load_data, build_resnet, train_model, plot_training_curves, plot_confusion_matrix, show_sample_predictions

def main():
    
    """
     解析命令行参数
    --data_dir：数据目录路径，默认为 ./COVID-19/COVID-19+Aug+CGAN。
    --image_size：图像大小，默认为 256。
    --batch_size：批量大小，默认为 8。
    --num_epochs：训练轮数，默认为 15。
    --lr：学习率，默认为 0.01。
    --device：训练设备，默认为 cuda（如果可用），否则为 cpu
    """
    
    parser = argparse.ArgumentParser(description="COVID-19 图像分类应用")
    parser.add_argument("--data_dir", type=str, default="./covid19_rawdata/COVID-19+Aug+CGAN", help="数据目录路径")
    parser.add_argument("--image_size", type=int, default=256, help="图像大小")
    parser.add_argument("--batch_size", type=int, default=8, help="批量大小")
    parser.add_argument("--num_epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")#如果有gpu就用gpu
    args = parser.parse_args()

    # 加载数据
    train_loader, val_loader, test_loader = load_data(args.data_dir, args.image_size, args.batch_size)

    # 构建模型
    net = build_resnet()

    # 训练模型
    print("开始训练模型...")
    train_losses, train_accs, val_accs = train_model(net, train_loader, val_loader, args.num_epochs, args.lr, args.device)

    # 可视化训练曲线
    print("绘制训练曲线...")
    plot_training_curves(train_losses, train_accs, val_accs)

    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    plot_confusion_matrix(net, test_loader, args.device)

    # 展示样本预测结果
    print("展示样本预测结果...")
    show_sample_predictions(net, test_loader, args.device)

    print("应用运行完成！")

if __name__ == "__main__":
    main()