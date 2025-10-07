import json
import re

from openai import OpenAI

from Neo4j_tools import update_group_node
from llm_core import call_LLM
from logger_config import setup_logger
from tools import read_jsonl_to_list, count_txt_files

from concurrent.futures import ThreadPoolExecutor, as_completed
# 日志文件路径
log_file_path = 'UniRAG.log'
# 设置日志记录器
logger = setup_logger(log_file_path)


def merge_fusion(filename):
    fusion_list = read_jsonl_to_list(f"Data/{filename}/{filename}_group_fusion.jsonl")
    for each in fusion_list:
        update_group_node(each)

    stay_list = read_jsonl_to_list(f"Data/{filename}/{filename}_group_stay.jsonl")
    for each in stay_list:
        update_group_node(each)



if __name__ == '__main__':
    print("开始执行step3-2-局部融合写进Neo4j")
    merge_fusion("gmzz_1")