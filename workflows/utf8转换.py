import os


def convert_ansi_to_utf8(file_path):
    # 读取ANSI格式的文件内容
    with open(file_path, 'r', encoding='ansi') as file:
        content = file.read()

    # 将内容以UTF-8格式写回原文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


if __name__ == "__main__":
    # 替换为你的文件路径
    file_path = 'Data/fr.txt'

    # 检查文件是否存在
    if os.path.exists(file_path):
        convert_ansi_to_utf8(file_path)
        print(f"文件 '{file_path}' 已成功转换为UTF-8格式并覆盖原文件。")
    else:
        print(f"文件 '{file_path}' 不存在，请检查路径。")