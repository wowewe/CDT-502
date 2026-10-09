#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
支持向量机 (SVM) 模型
"""

import os
import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import hinge_loss
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA


def extract_features(image_path):
    """
    从图像路径中提取特征
    
    参数:
        image_path: 图像文件路径
    
    返回:
        特征字典
    """
    # 从路径中提取类别
    category = os.path.basename(os.path.dirname(image_path))
    
    # 根据类别确定特征值：有裂分为 1（Positive），无裂分为 0（Negative）
    has_crack = 1 if category == 'Positive' else 0
    
    # 构建特征字典
    feature_dict = {'has_crack': has_crack}
    
    return feature_dict


def load_image_gray(image_path, size=(64, 64)):
    """
    读取并预处理图像为灰度图
    
    参数:
        image_path: 图像文件路径
        size: 目标尺寸
    
    返回:
        灰度图像数组
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    image = cv2.resize(image, size)
    return image.flatten()


def load_dataset(data_dir, load_images=False):
    """
    加载数据集
    
    参数:
        data_dir: 数据集目录路径
        load_images: 是否加载图像像素作为特征
    
    返回:
        features: 特征列表
        labels: 标签列表
    """
    features = []
    labels = []
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for image_file in os.listdir(class_dir):
                if image_file.endswith('.jpg') or image_file.endswith('.png'):
                    image_path = os.path.join(class_dir, image_file)
                    
                    if load_images:
                        img = load_image_gray(image_path)
                        if img is not None:
                            features.append({'pixels': img.tolist()})
                    else:
                        features.append(extract_features(image_path))
                    
                    labels.append(class_name)
    
    return features, labels


def apply_pca(features, n_components=30):
    """
    使用 PCA 进行降维
    
    参数:
        features: 特征列表
        n_components: 主成分数量
    
    返回:
        降维后的特征数组
    """
    # 将特征字典转换为向量
    vectorizer = DictVectorizer(sparse=False)
    features_encoded = vectorizer.fit_transform(features)
    
    # 应用 PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_encoded)
    
    return features_pca


def train_svm(features, labels, 
                kernel='linear', C=1.0,
                apply_pca_reduction=True):
    """
    训练 SVM 模型
    
    参数:
        features: 特征列表
        labels: 标签列表
        kernel: SVM 核函数
        C: 正则化参数
        apply_pca_reduction: 是否应用 PCA 降维
    
    返回:
        训练好的 SVM 分类器
    """
    # 编码标签
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # 转换为向量
    vectorizer = DictVectorizer(sparse=False)
    features_encoded = vectorizer.fit_transform(features)
    
    # 降维（可选）
    if apply_pca_reduction:
        pca = PCA(n_components=30)
        features_encoded = pca.fit_transform(features_encoded)
    
    # 训练 SVM
    svm_classifier = svm.SVC(kernel=kernel, C=C)
    svm_classifier.fit(features_encoded, labels_encoded)
    
    return svm_classifier, label_encoder, vectorizer


def evaluate_svm(svm_model, features, labels, 
                 label_encoder, vectorizer, apply_pca_reduction=True):
    """
    评估 SVM 模型
    
    参数:
        svm_model: 训练好的 SVM 模型
        features: 特征列表
        labels: 标签列表
        label_encoder: 标签编码器
        vectorizer: 特征向量器
        apply_pca_reduction: 是否应用 PCA 降维
    
    返回:
        评估指标字典
    """
    # 编码标签
    labels_encoded = label_encoder.transform(labels)
    
    # 转换为向量
    features_encoded = vectorizer.transform(features)
    
    # 降维（可选）
    if apply_pca_reduction:
        pca = PCA(n_components=30)
        features_encoded = pca.fit_transform(features_encoded)
    
    # 预测
    predictions = svm_model.predict(features_encoded)
    
    # 计算指标
    accuracy = accuracy_score(labels_encoded, predictions)
    precision = precision_score(labels_encoded, predictions, zero_division=0)
    recall = recall_score(labels_encoded, predictions, zero_division=0)
    f1 = f1_score(labels_encoded, predictions, zero_division=0)
    hinge_loss_value = hinge_loss(labels_encoded, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hinge_loss': hinge_loss_value
    }


def cross_validate_svm(features, labels, 
                        kernel='linear', C=1.0, 
                        n_splits=5, apply_pca_reduction=True):
    """
    使用交叉验证评估 SVM
    
    参数:
        features: 特征列表
        labels: 标签列表
        kernel: SVM 核函数
        C: 正则化参数
        n_splits: 交叉验证折数
        apply_pca_reduction: 是否应用 PCA 降维
    
    返回:
        交叉验证分数
    """
    # 编码标签
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # 转换为向量
    vectorizer = DictVectorizer(sparse=False)
    features_encoded = vectorizer.fit_transform(features)
    
    # 降维（可选）
    if apply_pca_reduction:
        pca = PCA(n_components=30)
        features_encoded = pca.fit_transform(features_encoded)
    
    # 交叉验证
    scores = cross_val_score(svm.SVC(kernel=kernel, C=C), 
                             features_encoded, labels_encoded, 
                             cv=n_splits)
    
    return scores


if __name__ == "__main__":
    print("=" * 50)
    print("开始训练 SVM 模型...")
    print("=" * 50)
    
    # 加载训练集
    train_dir = "dataset_split/train"
    train_features, train_labels = load_dataset(train_dir, load_images=True)
    print(f"Loaded {len(train_features)} training samples")
    
    # 训练模型
    svm_model, label_encoder, vectorizer = train_svm(
        train_features, train_labels, 
        kernel='linear', C=1.0
    )
    
    # 评估模型
    metrics = evaluate_svm(
        svm_model, train_features, train_labels,
        label_encoder, vectorizer
    )
    
    print("\nSVM 模型评估结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 交叉验证
    cv_scores = cross_validate_svm(train_features, train_labels, n_splits=5)
    print(f"\n5 折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
