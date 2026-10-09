# 裂纹图像识别智能平台

基于机器学习与深度学习的裂纹检测系统，支持多种模型（CNN、ResNet、SVM、KMeans）进行比较分析。

## 项目结构

```
E:\CDT502\
├── README.md                 # 项目文档
├── requirements.txt          # Python 依赖包
├── dataset_split.py          # 数据集划分模块（留出法/5 折交叉验证）
├── cnn_model.py              # 卷积神经网络模型
├── resnet_model.py           # 残差网络模型
├── svm_model.py              # 支持向量机模型
├── kmeans_model.py           # K-Means 聚类模型
└── dataset/                  # 原始数据集目录
```

## 功能模块

### 1. 数据集划分

支持两种划分方法：

- **留出法**: 80% 训练、10% 验证、10% 测试
- **5 折交叉验证**: 将数据集均分为 5 份，轮流作为验证集

使用示例：
```python
from dataset_split import split_dataset_leave_method, split_dataset_kfold

# 留出法
split_dataset_leave_method()

# 5 折交叉验证
split_dataset_kfold(k_num=5)
```

### 2. CNN 模型

使用 TensorFlow/Keras 构建的卷积神经网络，适合图像分类任务。

### 3. ResNet 模型

使用 PyTorch 实现的预训练 ResNet-18，迁移学习裂纹检测。

### 4. SVM 模型

使用 scikit-learn 实现的支持向量机，支持 PCA 降维和交叉验证。

### 5. KMeans 模型

无监督学习聚类，自动将图像分为有裂纹/无裂纹两类。

## 环境配置

### 系统要求

- Python 3.8 或更高版本
- Windows/Linux/macOS

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖包列表：

| 包名 | 最低版本 | 用途 |
|------|---------|------|
| numpy | 1.24.0 | 数值计算 |
| opencv-python | 4.8.0 | 图像处理 |
| tensorflow | 2.13.0 | 深度学习框架 |
| torch | 2.0.0 | 深度学习框架 |
| torchvision | 0.15.0 | 计算机视觉库 |
| scikit-learn | 1.3.0 | 机器学习算法 |
| pillow | 10.0.0 | 图像处理 |
| tqdm | 4.66.0 | 进度条 |
| ipywidgets | 8.0.0 | 交互式界面 |
| matplotlib | 3.7.0 | 数据可视化 |

### GPU 支持

如需使用 GPU 加速训练：

1. 安装 CUDA 11.8 和 cuDNN 8.x
2. 安装 tensorflow-gpu 或 torch 的 GPU 版本

## 使用指南

### 数据集准备

将原始数据集放入 `dataset/` 目录，目录结构：
```
dataset/
├── Positive/    # 有裂纹图片
└── Negative/    # 无裂纹图片
```

### 运行各模块

```bash
# 数据集划分
python dataset_split.py

# CNN 模型训练
python cnn_model.py

# ResNet 模型训练
python resnet_model.py

# SVM 模型训练
python svm_model.py

# KMeans 聚类
python kmeans_model.py
```

## 模型性能对比

| 模型 | 准确率 | 优点 | 缺点 |
|------|-------|------|------|
| CNN | 99.5% | 自动特征学习，适合图像任务 | 需要大量数据和计算资源 |
| ResNet | 99.6% | 深度网络，泛化能力强 | 模型复杂，训练时间长 |
| SVM | 97.5% | 小样本表现好，泛化能力强 | 大规模数据训练慢 |
| KMeans | 98.1% | 无需标签，实现简单 | 需要提前指定簇数 |

## 注意事项

1. **路径问题**: 所有路径已改为相对路径，确保数据集在 `dataset/` 目录下
2. **内存需求**: ResNet 模型需要至少 8GB 内存
3. **训练时间**: 深度学习模型训练可能需要数小时

## 许可证

本项目仅供CDT502学习和研究使用。
# CDT-502
