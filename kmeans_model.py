"""
K-Means 聚类模型
"""

import os
import shutil

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_images_from_folder(folder):
    """
    从文件夹加载所有图像
    
    参数:
        folder: 文件夹路径
    
    返回:
        images: 图像列表
        file_paths: 文件路径列表
    """
    images = []
    file_paths = []

    # 获取所有图像文件
    image_files = [
        filename for filename in os.listdir(folder)
        if filename.endswith('.jpg') or filename.endswith('.png')
    ]

    # 使用进度条
    for filename in tqdm(image_files, desc=f"Loading images from {os.path.basename(folder)}"):
        img_path = os.path.join(folder, filename)
        image = cv2.imread(img_path)

        if image is not None:
            images.append(image)
            file_paths.append(img_path)

    return images, file_paths


def preprocess_images(images):
    """
    预处理图像：调整大小并转换为灰度
    
    参数:
        images: 图像列表
    
    返回:
        preprocessed_images: 预处理后的图像列表
    """
    preprocessed_images = []

    for image in images:
        # 调整大小
        resized_image = cv2.resize(image, (100, 100))
        # 转换为灰度
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        # 展平
        flattened_image = gray_image.flatten()
        preprocessed_images.append(flattened_image)

    return preprocessed_images


def apply_pca(images, n_components=30):
    """
    使用 PCA 降维
    
    参数:
        images: 图像数组
        n_components: 主成分数量
    
    返回:
        降维后的图像数组
    """
    pca = PCA(n_components=n_components)
    pca_images = pca.fit_transform(images)
    return pca_images


def save_classified_images(paths, labels, output_folder):
    """
    保存分类后的图像
    
    参数:
        paths: 图像路径列表
        labels: 标签列表
        output_folder: 输出文件夹
    """
    os.makedirs(output_folder, exist_ok=True)
    for path, label in zip(paths, labels):
        filename = os.path.basename(path)
        output_path = os.path.join(output_folder, f"{label}", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copyfile(path, output_path)


def train_kmeans(data_dir="dataset", n_clusters=2, save_folder="Classified_Images"):
    """
    训练 K-Means 聚类模型
    
    参数:
        data_dir: 数据集目录
        n_clusters: 聚类簇数
        save_folder: 保存文件夹
    
    返回:
        cluster_labels: 聚类标签
    """
    all_images = []
    all_paths = []
    all_labels = []

    # 加载所有类别的图像
    for category in os.listdir(data_dir):
        category_folder = os.path.join(data_dir, category)
        if os.path.isdir(category_folder):
            images, paths = load_images_from_folder(category_folder)
            preprocessed_images = preprocess_images(images)

            all_images.extend(preprocessed_images)
            all_paths.extend(paths)
            all_labels.extend([category] * len(preprocessed_images))

    print(f"Loaded {len(all_images)} images total")

    # 标准化数据
    scaler = StandardScaler()
    scaled_images = scaler.fit_transform(all_images)

    # 应用 PCA 降维
    pca_images = apply_pca(scaled_images, n_components=30)

    # 训练 K-Means
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        max_iter=100,
        tol=1e-4,
        random_state=0
    )
    cluster_labels = kmeans.fit_predict(pca_images)

    # 保存分类后的图像
    for label in np.unique(cluster_labels):
        output_folder = os.path.join(save_folder, f"{label}")
        paths_batch = [
            path for path, lbl in zip(all_paths, cluster_labels)
            if lbl == label
        ]
        save_classified_images(paths_batch, [label] * len(paths_batch), output_folder)

    print("Images classified and saved.")

    return cluster_labels


def evaluate_kmeans(cluster_labels, true_labels):
    """
    评估 K-Means 聚类结果
    
    参数:
        cluster_labels: 聚类标签
        true_labels: 真实标签
    
    返回:
        准确性分数
    """
    from sklearn.metrics import adjusted_rand_score

    # 由于是无监督学习，使用 ARI 评估
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f"Adjusted Rand Index: {ari:.4f}")

    return ari


if __name__ == "__main__":
    print("=" * 50)
    print("开始训练 K-Means 聚类模型...")
    print("=" * 50)

    cluster_labels = train_kmeans()
    print(f"\n聚类完成，共 {len(cluster_labels)} 张图片被分类")
    print(f"簇分布: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")
