import os
import re
import shutil
from glob import glob
from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained("deepseek_tokenizer")


def split_novel_smart(input_path, output_dir, target_size=20000, tolerance=0.1):
    """
    智能分割小说文本，保持句子完整性
    参数：
    target_size: 目标字数 (默认20000)
    tolerance: 允许浮动比例 (默认±10%)
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 预定义分割参数
    min_size = int(target_size * (1 - tolerance))
    max_size = int(target_size * (1 + tolerance))

    sentences = sentence_splitter(text)
    print(f"切割好了，总共有个片段{len(sentences)}")
    # str to token
    encoded_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    print(f"编码好了，总共有个片段{len(encoded_sentences)}")

    chunk = []  # 当前正在构建的块
    current_size = 0  # 当前块的大小
    chunk_num = 1  # 记录当前块的编号

    for sentence, encoded_sentence in zip(sentences, encoded_sentences):
        encoded_sentence_size = len(encoded_sentence)  # 这里使用token的大小

        # 缓冲处理逻辑
        if current_size + encoded_sentence_size > max_size:
            # 寻找最佳分割点
            split_position = find_best_split("".join(chunk), current_size, min_size, max_size)

            if split_position < len("".join(chunk)):
                # 写入当前块前半部分
                write_chunk("".join(chunk)[:split_position], chunk_num, output_dir)
                print(f"当前写入时：currentSize为:{current_size}")
                # 剩余部分作为新块
                remaining = "".join(chunk)[split_position:]
                chunk = [remaining]
                current_size = len(remaining)
                chunk_num += 1
            else:
                write_chunk("".join(chunk), chunk_num, output_dir)
                print(f"当前写入时：currentSize为:{current_size}")
                chunk_num += 1
                chunk = []
                current_size = 0

        # 正常添加句子
        if encoded_sentence_size > max_size:
            # 处理超长句子
            chunk_num = handle_oversized_sentence(sentence, output_dir, chunk_num, target_size)
            print(f"超长句子长度为:{encoded_sentence_size}")
        else:
            chunk.append(sentence)
            current_size += encoded_sentence_size

    # 写入最后剩余内容
    if chunk:
        write_chunk("".join(chunk), chunk_num, output_dir)
        print(f"当前写入时：currentSize为:{current_size}")

def sentence_splitter(text):
    """增强型中英文分句器，优化处理英文缩写问题，并改为生成器"""
    # 处理包含对话的特殊情况（中文部分）
    dialog_pattern = r'(?<=[。！？…])(?=「|“|【|《|‘)'
    text = re.sub(dialog_pattern, '\n', text)

    # 英文缩写保护，避免误拆，似乎没用，因此注释掉，有特殊情况再处理
    # abbreviations = r'\b(?:Mr|Mrs|Dr|Jr|Sr|vs|etc|i\.e|e\.g|U\.S)\.'
    # text = re.sub(fr'({abbreviations})\s', r'\1<SPLIT>', text)

    # 句子结束的标记（中文和英文标点结合）
    sentence_endings = r'([。！？?！](?:」|”|）|》)?|(?<!\w\.\w)(?<![A-Z][a-z]\.)(?<=\w\.)\s+(?=[A-Z])|!|\"\'\）]|\?\")'

    # 使用 re.split 进行初步分割
    split_sentences = re.split(sentence_endings, text)

    return split_sentences


def find_best_split(chunk_text, current_size, min_size, max_size):
    """智能寻找最佳分割点，支持中英文混合"""
    # 优先寻找段落分隔
    last_paragraph = chunk_text.rfind('\n\n')
    if last_paragraph != -1 and last_paragraph > min_size:
        return last_paragraph + 2

    # 中英文句子结束标记，包含中文和英文的标点符号
    sentence_end_markers = ['。', '！', '？', '.', '!', '?', '…']

    # 查找最近的句子结束点
    for pos in range(len(chunk_text) - 1, min_size - 1, -1):
        if chunk_text[pos] in sentence_end_markers:
            # 对于英文文本，检查是否有后续的标点组合
            if pos < len(chunk_text) - 1 and chunk_text[pos + 1] in ['”', '’', '」']:  # 中文引号的组合
                return pos + 2
            # 如果是英文句号，可能后面会有引号、括号等符号
            if chunk_text[pos] == '.' and pos < len(chunk_text) - 1 and chunk_text[pos + 1] in ['"', "'", ')']:
                return pos + 2
            # 返回当前位置后一个字符的位置作为切割点
            return pos + 1

    # 如果没有找到合适的分割点，则强制在max_size处分割
    return min(len(chunk_text), max_size)


def handle_oversized_sentence(sentence, output_dir, chunk_num, target_size):
    """处理超长句子（超过max_size的1.2倍）"""
    print(f"发现超长句子，原文字数为：（{len(sentence)}字），启动特殊处理...")
    # 按逗号分割作为保底策略
    parts = re.split(r'([，,])', sentence)
    parts = [parts[i] + parts[i + 1] for i in range(0, len(parts) - 1, 2)]

    sub_chunk = []
    sub_size = 0
    for part in parts:
        if sub_size + len(part) > target_size:
            write_chunk("".join(sub_chunk), chunk_num, output_dir)
            chunk_num += 1
            sub_chunk = []
            sub_size = 0
        sub_chunk.append(part)
        sub_size += len(part)
    if sub_chunk:
        write_chunk("".join(sub_chunk), chunk_num, output_dir)
        chunk_num += 1
    return chunk_num


def write_chunk(content, chunk_num, output_dir):
    filename = os.path.join(output_dir, f'chunk_{chunk_num:03d}.txt')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'生成文件：{filename} [字数：{len(content)}]')


def split_twice(base_name: str):
    # 原始文件配置
    input_novel = f"Data/{base_name}.txt"

    # 目录配置
    temp_large_dir = "temp_large_chunks"
    final_output_dir = f"Data/Chunks/{base_name}_Small"

    # 第一次拆分：20k大块
    split_novel_smart(input_novel, temp_large_dir, target_size=20000)

    # 创建最终输出目录
    os.makedirs(final_output_dir, exist_ok=True)

    # 处理每个大块文件
    for large_chunk in sorted(glob(os.path.join(temp_large_dir, "chunk_*.txt")),
                              key=lambda x: int(re.findall(r"\d+", x)[0])):
        # 提取大块序号
        large_num = re.search(r"chunk_(\d+).txt", large_chunk).group(1)

        # 临时小块目录
        temp_small_dir = f"temp_small_{large_num}"

        # 第二次拆分：800字小块
        split_novel_smart(large_chunk, temp_small_dir, target_size=800)

        # 处理每个小块文件
        for small_chunk in sorted(glob(os.path.join(temp_small_dir, "chunk_*.txt")),
                                  key=lambda x: int(re.findall(r"\d+", x)[0])):
            # 提取小块序号
            small_num = re.search(r"chunk_(\d+).txt", small_chunk).group(1)

            # 构建新文件名
            new_name = f"{base_name}_{large_num}_{small_num}.txt"
            new_path = os.path.join(final_output_dir, new_name)

            # 移动并重命名文件
            shutil.move(small_chunk, new_path)
            print(f"生成最终文件：{new_path}")

        # 清理临时目录
        shutil.rmtree(temp_small_dir)

    # 清理大块临时目录
    shutil.rmtree(temp_large_dir)


def check_dir(filename):
    # 指定文件夹路径
    folder_path = f"Data/{filename}"

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 如果不存在，则创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已存在。")


def check_file(filename):
    file_path = f"Data/{filename}.txt"
    if os.path.exists(file_path):
        print(f"文件 '{file_path}' 存在。")
        return True
    else:
        print(f"文件 '{file_path}' 不存在。")
        return False


def main0_0(filename):
    if not check_file(filename):
        return
    check_dir(filename)
    # 使用示例
    # split_novel_smart(f"Data/{filename}.txt", f"Data/Chunks/{filename}_Large", target_size=20000)
    split_twice(filename)


if __name__ == '__main__':
    filename = "gmzz_1"
    main0_0(filename)
