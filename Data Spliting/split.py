import random
def split_dataset(file_path, train_ratio=0.7, validation_ratio=0.2):
    # 读取数据集
    with open(file_path, 'r') as f:
        data = f.readlines()
    # 随机打乱数据集
    random.shuffle(data)
    # 计算训练集和测试集的边界
    train_size = int(len(data) * train_ratio)
    validation_size = int(len(data) * validation_ratio)
    test_size = len(data) - train_size - validation_size
    # 划分训练集和测试集
    train_set = data[: train_size]
    validate_set = data[train_size: train_size + validation_size]
    test_set = data[test_size + validation_size:]
    # 保存训练集和测试集到对应的txt文件
    with open('Aug_train_set.txt', 'w') as f:
        f.writelines(train_set)
    with open('Aug_validation_set.txt', 'w') as f:
        f.writelines(validate_set)
    with open('Aug_test_set.txt', 'w') as f:
        f.writelines(test_set)
    print(f"数据集已成功划分为训练集、验证集和测试集，并保存到对应的txt文件中。")


# 调用函数，以7:2:1的比例划分训练集和测试集
split_dataset('output.txt',0.7, 0.2)