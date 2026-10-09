#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能平台实验 - 交互式裂纹检测界面
"""

import glob
import os
import warnings

import numpy as np
from PIL import Image
from keras.utils import to_categorical
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')

# 设置数据目录
data_dir = "dataset_split/train"


def load_images(data_dir, img_size=(64, 64)):
    """
    读取图片数据
    
    参数:
        data_dir: 数据目录
        img_size: 图像尺寸
    
    返回:
        images: 图像数据
        labels: 标签
    """
    images = []
    labels = []

    for folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, folder)
        if os.path.isdir(class_path):
            for img_path in glob.glob(os.path.join(class_path, '*.jpg')):
                try:
                    img = Image.open(img_path).resize(img_size)
                    img = np.array(img)

                    if img.shape == (img_size[0], img_size[1], 3):
                        images.append(img)
                        labels.append(folder)
                except Exception:
                    continue

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def main():
    """
    主函数：创建交互式界面
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("错误：需要安装 ipywidgets")
        print("运行: pip install ipywidgets")
        return

    # 数据划分选项
    data_split_method = widgets.Dropdown(
        options=['留出法', '5 折交叉验证'],
        value='留出法',
        description='数据划分方法:',
    )

    # 模型选择
    model_choice = widgets.Dropdown(
        options=['SVM', 'CNN', 'K-means', 'ResNet'],
        value='SVM',
        description='选择模型:',
    )

    # 损失函数选择
    def update_loss_function_options(change):
        if change.new == 'SVM':
            loss_function.options = ['Binary Crossentropy', 'Hinge Loss']
        elif change.new == 'CNN':
            loss_function.options = ['Categorical Crossentropy', 'Binary Crossentropy']
        elif change.new == 'K-means':
            loss_function.options = ['K-means Loss']
        elif change.new == 'ResNet':
            loss_function.options = ['ResNet Loss']

    model_choice.observe(update_loss_function_options, names='value')

    loss_function = widgets.Dropdown(description='损失函数:')

    # 超参数设置 - 学习率
    learning_rate = widgets.FloatSlider(
        value=0.01,
        min=0.001,
        max=0.1,
        step=0.001,
        description='学习率:',
    )

    # 训练按钮
    train_button = widgets.Button(description="开始训练")

    # 结果输出
    output = widgets.Output()

    def train_model(b):
        with output:
            output.clear_output()

            try:
                print("加载数据...")
                images, labels = load_images(data_dir)

                if len(images) == 0:
                    print("错误：未找到任何图像。请检查数据集路径。")
                    return

                print(f"加载了 {len(images)} 张图片")

                # 数据划分
                if data_split_method.value == '留出法':
                    X_train, X_test, y_train, y_test = train_test_split(
                        images, labels, test_size=0.2, random_state=42
                    )
                else:
                    kf = KFold(n_splits=5)
                    for train_index, test_index in kf.split(images):
                        X_train, X_test = images[train_index], images[test_index]
                        y_train, y_test = labels[train_index], labels[test_index]

                # 标签编码
                if isinstance(y_train[0], str):
                    label_encoder = LabelEncoder()
                    y_train = label_encoder.fit_transform(y_train)
                    y_test = label_encoder.transform(y_test)

                # 模型训练
                if model_choice.value == 'SVM':
                    print("训练 SVM 模型...")
                    X_train_flat = X_train.reshape(X_train.shape[0], -1)
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)

                    model = SVC(kernel='linear')
                    model.fit(X_train_flat, y_train)

                    y_pred_test = model.predict(X_test_flat)
                    accuracy_test = accuracy_score(y_test, y_pred_test)

                    if loss_function.value == 'Hinge Loss':
                        h_loss_test = np.maximum(0, 1 - y_test * y_pred_test)
                        average_h_loss_test = np.mean(h_loss_test)
                        print(f"测试集准确率: {accuracy_test:.4f}")
                        print(f"平均合页损失: {average_h_loss_test:.4f}")
                    else:
                        print(f"测试集准确率: {accuracy_test:.4f}")

                elif model_choice.value == 'CNN':
                    print("构建和训练 CNN 模型...")
                    num_classes = len(np.unique(y_train))

                    model = Sequential([
                        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
                        Conv2D(64, (3, 3), activation='relu'),
                        Flatten(),
                        Dense(64, activation='relu'),
                        Dense(num_classes, activation='softmax')
                    ])

                    model.compile(
                        optimizer=Adam(learning_rate=learning_rate.value),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )

                    y_train_encoded = to_categorical(y_train, num_classes=num_classes)
                    y_test_encoded = to_categorical(y_test, num_classes=num_classes)

                    history = model.fit(
                        X_train, y_train_encoded,
                        epochs=10,
                        validation_data=(X_test, y_test_encoded),
                        verbose=1
                    )

                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(
                        np.argmax(y_test_encoded, axis=1),
                        np.argmax(y_pred, axis=1)
                    )
                    precision = precision_score(
                        np.argmax(y_test_encoded, axis=1),
                        np.argmax(y_pred, axis=1),
                        average='weighted'
                    )
                    recall = recall_score(
                        np.argmax(y_test_encoded, axis=1),
                        np.argmax(y_pred, axis=1),
                        average='weighted'
                    )
                    f1 = f1_score(
                        np.argmax(y_test_encoded, axis=1),
                        np.argmax(y_pred, axis=1),
                        average='weighted'
                    )

                    print("\n训练完成!")
                    print(f"测试集准确率: {accuracy:.4f}")
                    print(f"精确度: {precision:.4f}")
                    print(f"召回率: {recall:.4f}")
                    print(f"F1 分数: {f1:.4f}")

                elif model_choice.value == 'K-means':
                    print("训练 K-Means 模型...")
                    X_train_flat = X_train.reshape(X_train.shape[0], -1)
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)

                    model = KMeans(n_clusters=len(np.unique(y_train)), random_state=0)
                    model.fit(X_train_flat)

                    y_pred = model.predict(X_test_flat)
                    accuracy = accuracy_score(y_test, y_pred)

                    print(f"\n测试集准确率: {accuracy:.4f}")

                else:
                    print("ResNet 模型需要使用 resnet_model.py 单独运行")

            except Exception as e:
                print(f"错误: {e}")

    train_button.on_click(train_model)

    # 界面布局
    print("=" * 50)
    print("裂纹检测智能平台")
    print("=" * 50)
    display(data_split_method, model_choice, loss_function, learning_rate, train_button, output)


if __name__ == "__main__":
    main()
