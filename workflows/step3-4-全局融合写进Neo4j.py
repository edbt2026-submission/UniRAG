import json
import re

from openai import OpenAI

from Neo4j_tools import update_group_node, create_global
from llm_core import call_LLM
from logger_config import setup_logger
from tools import read_jsonl_to_list, count_txt_files

from concurrent.futures import ThreadPoolExecutor, as_completed
# 日志文件路径
log_file_path = 'UniRAG.log'
# 设置日志记录器
logger = setup_logger(log_file_path)


def merge_fusion_global(filename):
    fusion_list = read_jsonl_to_list(f"Data/{filename}/{filename}_group_after_global.jsonl")
    for g in fusion_list:
        create_global(g["global_id"], g["global_name"], g["global_description"], g["group_ids"], g["source_index"], g["other_names"])



if __name__ == '__main__':
    print("开始执行step3-4-全局融合写进Neo4j")
    merge_fusion_global("gmzz_1")