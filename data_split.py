import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.2

    # 指向你的flower_photos文件夹
    # cwd = os.getcwd()
    data_root = "F:/CodeForFuture/data/"
    origin_sar_path = os.path.join(data_root, "imagedata")
    assert os.path.exists(origin_sar_path), "path '{}' does not exist.".format(origin_sar_path)

    sar_class = [cls for cls in os.listdir(origin_sar_path)
                    if os.path.isdir(os.path.join(origin_sar_path, cls))]

    # 建立保存训练集的文件夹
    train_root = os.path.join(data_root + 'SAR_data/', "train")
    mk_file(train_root)
    for cls in sar_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cls))

    # 建立保存验证集的文件夹
    val_root = os.path.join(data_root + 'SAR_data/', "val")
    mk_file(val_root)
    for cls in sar_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cls))

    for cls in sar_class:
        cla_path = os.path.join(origin_sar_path, cls)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cls)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cls)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cls, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()