"""
数据集划分模块
包含留出法和 5 折交叉验证两种数据集划分方法
"""

import os
import random
import shutil

from sklearn.model_selection import KFold


def makedir(new_dir):
    """创建目录，如果已存在则不操作"""
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def split_dataset_leave_method(dataset_dir="dataset", split_dir="dataset_split", train_pct=0.8, valid_pct=0.1,
                               test_pct=0.1):
    """
    留出法数据集划分
    
    参数:
        dataset_dir: 原始数据集路径
        split_dir: 划分后保存路径
        train_pct: 训练集比例
        valid_pct: 验证集比例
        test_pct: 测试集比例
    """
    # 设置随机种子
    random.seed(1)

    # 确定路径
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "val")
    test_dir = os.path.join(split_dir, "test")

    # 创建目标目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历数据集文件夹
    for root, dirs, files in os.walk(dataset_dir):
        for sub_dir in dirs:
            # 获取当前类别文件夹下的所有图像文件
            imgs = os.listdir(os.path.join(root, sub_dir))
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            random.shuffle(imgs)

            img_count = len(imgs)
            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))

            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                os.makedirs(out_dir, exist_ok=True)
                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(root, sub_dir, imgs[i])
                shutil.copy(src_path, target_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point - train_point,
                                                                 img_count - valid_point))


def split_dataset_kfold(dataset_dir="dataset", save_path="dataset_kfold", k_num=5):
    """
    5 折交叉验证数据集划分
    
    参数:
        dataset_dir: 原始数据集路径
        save_path: 保存划分后的数据集路径
        k_num: 交叉验证的折数
    """
    # 创建保存目录
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    print("succeed:", save_path)

    # 定义交叉验证
    kf = KFold(n_splits=k_num, shuffle=True, random_state=0)

    file = os.listdir(dataset_dir)

    for i in file:
        root_sub = os.path.join(dataset_dir, i)

        if os.path.isdir(root_sub):
            file_sub = os.listdir(root_sub)
            print(f"Processing: {i}")

            for n, data in enumerate(kf.split(file_sub)):
                train, test = data
                save_path_sub_train = os.path.join(save_path, str(n), "train", i)
                save_path_sub_test = os.path.join(save_path, str(n), "test", i)

                os.makedirs(save_path_sub_train, exist_ok=True)
                os.makedirs(save_path_sub_test, exist_ok=True)

                print(f"Fold {n}: train={len(train)}, test={len(test)}")

                for item in train:
                    img_name = file_sub[item]
                    path_train = os.path.join(root_sub, img_name)
                    targe_path = os.path.join(save_path_sub_train, img_name)
                    shutil.copy(path_train, targe_path)

                for item in test:
                    img_name = file_sub[item]
                    path_train = os.path.join(root_sub, img_name)
                    targe_path = os.path.join(save_path_sub_test, img_name)
                    shutil.copy(path_train, targe_path)

    print("End...{}折交叉验证已完成, 数据集路径为: {}".format(k_num, save_path))


if __name__ == "__main__":
    # 留出法示例
    print("=" * 50)
    print("使用留出法划分数据集...")
    print("=" * 50)
    split_dataset_leave_method()

    # 5 折交叉验证示例
    print("\n" + "=" * 50)
    print("使用 5 折交叉验证划分数据集...")
    print("=" * 50)
    split_dataset_kfold()
