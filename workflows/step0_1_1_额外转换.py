import json
import re

from openai import OpenAI

from step0_1_0_Chunks_analysis import main0_0
from llm_core import call_LLM
from logger_config import setup_logger
from tools import read_jsonl_to_list, count_txt_files, reset_jsonl_file, write_jsonl_file, count_lines

from concurrent.futures import ThreadPoolExecutor, as_completed
# 日志文件路径
log_file_path = 'UniRAG.log'
# 设置日志记录器
logger = setup_logger(log_file_path)


def validate_entity(entity):
    required_keys = {"name", "type", "description"}
    if not required_keys.issubset(entity.keys()):
        return False
    return True


def validate_relationship(relationship):
    required_keys = {"source", "target", "description", "keywords", "strength"}
    if not required_keys.issubset(relationship.keys()):
        return False
    if not isinstance(relationship["keywords"], list):
        return False
    return True


def validate_keywords(keywords):
    if not isinstance(keywords, list):
        return False
    return True

def ex_change(filename):
    extract_result = read_jsonl_to_list(f"Data/{filename}/{filename}_extract_result.jsonl")
    right = 0
    wrong = 0
    right_list = []
    for er in extract_result:
        json_str = er['result']
        # 判断json_str类型为str还是dict
        if isinstance(json_str, str):
            # 尝试解析json
            try:
                json_data = json.loads(json_str)
                right += 1
                er['result'] = json_data
                right_list.append(er)
            except json.JSONDecodeError:
                wrong += 1
                print(f"{filename} 文件中，存在错误 JSON 数据：\n{json_str}")
        else:
            # 如果是dict类型，则直接使用
            right += 1
            right_list.append(er)

    # 清空原jsonl文件
    reset_jsonl_file(f"Data/{filename}/{filename}_extract_result.jsonl")

    # 重新写入
    write_jsonl_file(f"Data/{filename}/{filename}_extract_result.jsonl", right_list)



    # 输出统计结果
    print(f"{filename} 文件中，正确解析的 JSON 数据数量为：{right} 错误的 JSON 数据数量为：{wrong}")

    print("清理和转换完毕")
    return right


def check_json_format(filename):
    extract_result = read_jsonl_to_list(f"Data/{filename}/{filename}_extract_result.jsonl")
    right = 0
    wrong = 0
    right_list = []
    for er in extract_result:
        flag = 0
        json_list = er['result']
        for each in json_list:
            if "entity" in each:
                if not validate_entity(each["entity"]):
                    flag = 1
                    print("结点出错：")
                    print(json.dumps(each, ensure_ascii=False))
            elif "relationship" in each:
                if not validate_relationship(each["relationship"]):
                    flag = 1
                    print("关系出错：")
                    print(json.dumps(each, ensure_ascii=False))
            elif "keywords" in each:
                if not validate_keywords(each["keywords"]):
                    flag = 1
                    print("关键词出错：")
                    print(json.dumps(each, ensure_ascii=False))
        if flag == 0:
            right += 1
            right_list.append(er)
        else:
            wrong += 1
    reset_jsonl_file(f"Data/{filename}/{filename}_extract_result.jsonl")
    write_jsonl_file(f"Data/{filename}/{filename}_extract_result.jsonl", right_list)
    print(f"{filename} 文件中，格式正确的数量为：{right} 错误的 JSON 数据数量为：{wrong}")
    print("格式检查完毕")
    return right


if __name__ == '__main__':

    LLM_type = "Silicon"
    model_name = "Pro/deepseek-ai/DeepSeek-V3"
    worker = 3
    json_mode = False
    print("开始执行step-0-1-1-额外转换")
    filename = "gmzz_1"
    ex_change(filename)


    count_all = count_lines(f"Data/{filename}/{filename}_extract_result.jsonl")
    count_processed = 0

    analysis_list = read_jsonl_to_list(f"Data/{filename}/{filename}_text_analysis.jsonl")

    while count_processed < count_all:
        print("正在修补错误。。。")
        count_processed = check_json_format(filename)
        if count_processed < count_all:
            main0_0(filename, LLM_type, model_name, json_mode, worker)
