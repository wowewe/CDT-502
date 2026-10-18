"""
卷积神经网络 (CNN) 模型
"""

import os

import cv2
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential


def read_image(image_path, size=(28, 28)):
    """
    读取并预处理图像
    
    参数:
        image_path: 图像文件路径
        size: 目标尺寸，默认为 (28, 28)
    
    返回:
        处理后的图像数组
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    # 颜色通道转换：BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 调整大小
    image = cv2.resize(image, size)
    # 转换为 float 类型并归一化
    image = image.astype('float32') / 255.0

    return image


def load_dataset(data_dir, img_size=(28, 28)):
    """
    加载数据集
    
    参数:
        data_dir: 数据集目录路径
        img_size: 图像尺寸
    
    返回:
        X: 图像数据
        y: 标签数据
    """
    images = []
    labels = []
    label_encoder = {
        'Positive': 1,
        'Negative': 0
    }

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    img_path = os.path.join(class_path, img_file)
                    img = read_image(img_path, img_size)
                    if img is not None:
                        images.append(img)
                        labels.append(label_encoder.get(class_name, 0))

    return np.array(images), np.array(labels)


def build_model(input_shape=(28, 28, 3), num_classes=1):
    """
    构建 CNN 模型
    
    参数:
        input_shape: 输入图像形状
        num_classes: 输出类别数
    
    返回:
        编译好的 Keras 模型
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32, save_path='my_model.keras'):
    """
    训练模型
    
    参数:
        model: Keras 模型
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        epochs: 训练轮数
        batch_size: 批次大小
        save_path: 模型保存路径
    """
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'\nTest accuracy: {test_acc:.4f}')

    # 保存模型
    model.save(save_path)
    print(f'Model saved to {save_path}')

    return history, test_acc


if __name__ == "__main__":
    # 示例：加载数据并训练模型
    print("=" * 50)
    print("开始训练 CNN 模型...")
    print("=" * 50)

    # 加载数据集
    data_dir = "dataset_split/train"
    X_train, y_train = load_dataset(data_dir)

    test_dir = "dataset_split/test"
    X_test, y_test = load_dataset(test_dir)

    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    # 构建模型
    model = build_model()
    model.summary()

    # 训练模型
    history, accuracy = train_model(model, X_train, y_train, X_test, y_test)
