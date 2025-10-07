import json

from tools import reset_jsonl_file
from logger_config import setup_logger
# 日志文件路径
log_file_path = 'UniRAG.log'
# 设置日志记录器
logger = setup_logger(log_file_path)

def sort_jsonl_file(input_file_path, output_file_path):
    # 读取文件并解析每一行为JSON对象
    with open(input_file_path, 'r', encoding='utf-8') as file:
        # 使用生成器表达式来加载每一行的JSON
        json_list = (json.loads(line) for line in file)

        # 按照index_L和index_S排序
        sorted_json_list = sorted(json_list, key=lambda x: (x['index_L'], x['index_S']))

    # 将排序后的JSON对象写入新的JSONL文件
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for json_obj in sorted_json_list:
            # 将JSON对象转换为字符串并写入文件，每个对象占一行
            file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    textname = 'gmzz_1'
    # 指定输入和输出文件路径
    input_file_path = f'Data/{textname}/{textname}_extract_result.jsonl'
    output_file_path = f'Data/{textname}/{textname}_extract_result_s.jsonl'

    reset_jsonl_file(output_file_path)

    # 调用函数进行排序和写入
    sort_jsonl_file(input_file_path, output_file_path)

    with open(output_file_path, 'r', encoding='utf-8') as file:
        json_list = [json.loads(line) for line in file]

    ex_file = f'Data/{textname}/{textname}_EX.jsonl'
    reset_jsonl_file(ex_file)

    index = 1

    for json_obj in json_list:
        entities_list = []
        inner_index = 1
        for entity in json_obj['result']:
            if 'entity' not in entity:
                entities_list.append(entity)
                continue
            entity["inner_index"] = inner_index
            inner_index += 1
            entities_list.append(entity)

        new_entry = {"index": index, "index_L": json_obj['index_L'], "index_S": json_obj['index_S'], "EX": entities_list}
        with open(ex_file, 'a', encoding='utf-8') as file:
            file.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
        index += 1


