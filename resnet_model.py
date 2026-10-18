"""
残差网络 (ResNet) 模型
"""

import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models


class CracksDataset(Dataset):
    """裂痕图像数据集类"""

    def __init__(self, data_folders, transform=None):
        """
        参数:
            data_folders: 包含图像数据的文件夹路径列表
            transform: 数据预处理和增强
        """
        self.data_folders = data_folders
        self.transform = transform
        self.images = []
        self.labels = []

        # 加载所有图像
        for folder_idx, folder in enumerate(data_folders):
            if os.path.exists(folder):
                for class_name in os.listdir(folder):
                    class_path = os.path.join(folder, class_name)
                    if os.path.isdir(class_path):
                        for img_file in os.listdir(class_path):
                            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                                img_path = os.path.join(class_path, img_file)
                                self.images.append(img_path)
                                self.labels.append(1 if class_name == 'Positive' else 0)

        print(f"Loaded {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label


def get_dataloaders(data_folder, batch_size=32):
    """
    创建数据加载器
    
    参数:
        data_folder: 数据集目录
        batch_size: 批次大小
    
    返回:
        train_loader, test_loader
    """
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # 创建数据集
    dataset = CracksDataset([data_folder], transform=transform)

    # 分割为训练集和测试集
    from torch.utils.data import random_split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def build_resnet_model(pretrained=True):
    """
    构建 ResNet-18 模型
    
    参数:
        pretrained: 是否使用预训练权重
    
    返回:
        修改后的 ResNet 模型
    """
    # 加载预训练的 ResNet-18
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    resnet = models.resnet18(weights=weights)

    # 修改最后一层全连接层用于二分类
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, 2)

    return resnet


def train_resnet(model, train_loader, num_epochs=10, lr=0.001):
    """
    训练 ResNet 模型
    
    参数:
        model: ResNet 模型
        train_loader: 训练数据加载器
        num_epochs: 训练轮数
        lr: 学习率
    
    返回:
        训练好的模型
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model


def save_model(model, path='crack_detection_model.pth'):
    """保存训练好的模型"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    print("=" * 50)
    print("开始训练 ResNet 模型...")
    print("=" * 50)

    # 加载数据
    train_loader, test_loader = get_dataloaders("dataset_split/train")
    print(f"Training batches: {len(train_loader)}")

    # 构建模型
    model = build_resnet_model()

    # 训练模型
    model = train_resnet(model, train_loader)

    # 保存模型
    save_model(model)
