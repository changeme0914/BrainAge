import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
import Data
import GraphNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def main():
    # 实验设置
    model_path = './trained/GCN'  # 新模型保存路径
    data_path = 'D:\Data\derivatives\R2SN'  # 确保此处为64维特征数据（单模态MRI）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 64  # 目标输入维度
    epochs = 100
    lr = 1e-4
    val_split = 0.2  # 验证集比例
    batch_size = 32

    # 创建模型保存目录
    os.makedirs(model_path, exist_ok=True)

    # 数据准备与划分
    logging.info('Prepare data...')
    dataset = Data.Brain_network(data_path)
    # 划分训练集和验证集
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 数据加载器
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    # 网络构建（输入维度64）
    logging.info('Initialize network...')
    net = GraphNet.GraphNet(input_dim=input_dim)
    logging.info(f'  + Number of Model params: {sum([p.data.nelement() for p in net.parameters()])}')
    net = net.to(device)

    # 损失函数（MSE+交叉熵，交叉熵需将年龄离散化）
    mse_loss = nn.MSELoss()

    # 交叉熵损失需要将连续年龄转换为离散类别（示例：按10年间隔划分0-100岁为10类）
    def age_to_class(age, num_classes=10):
        return torch.clamp(torch.floor(age / 10).long(), 0, num_classes - 1)

    ce_loss = nn.CrossEntropyLoss()
    num_classes = 10  # 年龄类别数

    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 训练循环
    best_val_loss = float('inf')
    logging.info('Start training...')
    for epoch in range(1, epochs + 1):
        # 训练阶段
        net.train()
        train_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            output = net(data)  # 回归输出（连续年龄）
            age_pred = output.squeeze()
            age_true = data.y.float()

            # 计算MSE损失
            loss_mse = mse_loss(age_pred, age_true)

            # 计算交叉熵损失（需转换为类别）
            class_pred = torch.cat([age_pred.unsqueeze(1) for _ in range(num_classes)], dim=1)  # 适配交叉熵输入格式
            class_true = age_to_class(age_true, num_classes)
            loss_ce = ce_loss(class_pred, class_true)

            # 总损失（加权求和）
            loss = 0.7 * loss_mse + 0.3 * loss_ce  # 可调整权重

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs  # 累计损失

        train_loss /= len(train_loader.dataset)

        # 验证阶段
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = net(data).squeeze()
                age_true = data.y.float()

                # 计算验证损失
                loss_mse_val = mse_loss(output, age_true)
                class_pred_val = torch.cat([output.unsqueeze(1) for _ in range(num_classes)], dim=1)
                class_true_val = age_to_class(age_true, num_classes)
                loss_ce_val = ce_loss(class_pred_val, class_true_val)
                loss_val = 0.7 * loss_mse_val + 0.3 * loss_ce_val

                val_loss += loss_val.item() * data.num_graphs

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)  # 学习率调整

        # 打印日志
        logging.info(f'Epoch {epoch}/{epochs}')
        logging.info(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), os.path.join(model_path, 'model_GraphTransformer_64dim.pth'))
            logging.info(f'Saved best model (Val Loss: {best_val_loss:.4f})')

    logging.info('Training completed!')


if __name__ == '__main__':
    main()