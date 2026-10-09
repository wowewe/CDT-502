"""K-Means 聚类模型（优化版）
减少 CPU 和内存占用
"""

import os
import numpy as np
import cv2
import shutil
import gc
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler


def load_image_as_feature(image_path, size=(32, 32)):
    """
    逐张图片加载并预处理（节省内存）
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 缩小尺寸并转灰度
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten().astype('float32') / 255.0
    
    return img


def get_image_paths(data_dir, max_images_per_class=5000):
    """
    获取图像路径（限制数量）
    """
    image_paths = []
    class_names = []
    
    for category in os.listdir(data_dir):
        class_folder = os.path.join(data_dir, category)
        if not os.path.isdir(class_folder):
            continue
        
        imgs = [f for f in os.listdir(class_folder) if f.endswith(('.jpg', '.png'))]
        # 限制每个类别的图片数量（减少计算量）
        if len(imgs) > max_images_per_class:
            import random
            random.shuffle(imgs)
            imgs = imgs[:max_images_per_class]
        
        for img_file in imgs:
            image_paths.append(os.path.join(class_folder, img_file))
            class_names.append(category)
    
    return image_paths, class_names


def train_kmeans_optimized(data_dir="dataset", 
                           n_clusters=2, 
                           max_images_per_class=5000,
                           save_folder="Classified_Images"):
    """
    优化的 K-Means 训练
    """
    print("正在获取图像路径（限制数量以减少内存）...")
    image_paths, class_names = get_image_paths(data_dir, max_images_per_class)
    print(f"处理图像总数: {len(image_paths)} 张")
    
    # 分批加载和处理
    batch_size = 500
    features = []
    original_paths = []
    
    print("正在批量加载和预处理图像...")
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_features = []
        
        for path in batch_paths:
            feat = load_image_as_feature(path)
            if feat is not None:
                batch_features.append(feat)
                original_paths.append(path)
        
        features.extend(batch_features)
        
        # 每处理 10000 张释放内存
        if (i // 10000) * 10000 == i and i > 0:
            gc.collect()
            print(f"已处理 {i} 张...")
    
    features = np.array(features)
    print(f"特征矩阵形状: {features.shape}")
    
    # 标准化
    print("正在标准化...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 使用增量 PCA 减少维度（节省内存）
    print("正在降维...")
    n_components = min(50, features_scaled.shape[1] - 1)
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    features_pca = pca.fit_transform(features_scaled)
    
    # 训练 K-Means
    print(f"正在训练 K-Means (n_clusters={n_clusters})...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=256,  # 增大批次，加快训练
        max_iter=100,
        tol=1e-4,
        random_state=42,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(features_pca)
    
    # 保存结果
    print("正在保存分类结果...")
    os.makedirs(save_folder, exist_ok=True)
    
    # 映射回原始图像
    for path, label in zip(original_paths, cluster_labels):
        filename = os.path.basename(path)
        output_dir = os.path.join(save_folder, f"cluster_{label}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        shutil.copy2(path, output_path)
    
    # 清理内存
    del features, features_scaled, features_pca, cluster_labels
    gc.collect()
    
    print("\n完成!")
    print(f"聚类簇数: {n_clusters}")
    print(f"聚类分布: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")


if __name__ == "__main__":
    print("=" * 50)
    print("开始训练 K-Means 聚类模型（优化版）...")
    print("=" * 50)
    train_kmeans_optimized()
