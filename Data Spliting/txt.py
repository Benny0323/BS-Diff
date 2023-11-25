import os


def full_traverse_directory(directory):  # JSRT\JPCLN001.png
    # 创建一个空的列表用于存储文件名
    file_names = []
    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 获取完整的文件路径
            full_path = os.path.join(root, file)
            file_names.append(full_path)
    # 按照文件名排序（如果你希望的话）
    file_names.sort()
    # 创建一个新的txt文件，并将文件名写入该文件
    with open('output.txt', 'w') as f:
        for file_name in file_names:
            f.write(file_name + '\n')


def traverse_directory(directory):  # JPCLN001.png
    # 创建一个空的列表用于存储文件名
    file_names = []
    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(file)
    # 按照文件名排序（如果你希望的话）
    file_names.sort()
    # 创建一个新的txt文件，并将文件名写入该文件
    with open('output.txt', 'w') as f:
        for file_name in file_names:
            f.write(file_name + '\n')


if __name__ == "__main__":
    # 使用你希望遍历的目录替换'your_directory'
    traverse_directory('./sztumor')
