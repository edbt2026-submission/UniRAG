import uuid

from openai import OpenAI

from llm_core import call_LLM
from logger_config import setup_logger
from tools import read_jsonl_to_list, count_txt_files, write_jsonl_file

from concurrent.futures import ThreadPoolExecutor, as_completed
# 日志文件路径
log_file_path = 'UniRAG.log'
# 设置日志记录器
logger = setup_logger(log_file_path)


def entities_fusion(filename):
    ex_list = read_jsonl_to_list(f"Data/{filename}/{filename}_EX.jsonl")
    final_merged_list = read_jsonl_to_list(f"Data/{filename}/{filename}_final_merge.jsonl")

    if len(ex_list) != len(final_merged_list) + 1:
        logger.error("ex_list和final_merged_list长度不匹配，请检查！")
        return

    # 列表第一个元素，初始化
    for element in ex_list[0]["EX"]:
        if "entity" in element:
            entity_id = "entity_" + str(uuid.uuid4())
            group_id = "group_" + str(uuid.uuid4())
            element['entity_id'] = entity_id
            element['group_id'] = group_id
        else:
            relation_id = "relation_" + str(uuid.uuid4())
            element['relation_id'] = relation_id

    total_index = len(final_merged_list)

    # 列表剩余元素
    for index in range(0, total_index):

        # 当前列表
        current_list = ex_list[index + 1]["EX"]

        # 前一组列表
        previous_list = ex_list[index]["EX"]

        # 合并列表
        merge_list = final_merged_list[index]["Merge"]

        for element in current_list:
            if "entity" in element:
                entity_id = "entity_" + str(uuid.uuid4())
                element['entity_id'] = entity_id
                inner_index = element["inner_index"]
                p = check_group_id(inner_index, merge_list)
                if p != 0:
                    for each in previous_list:
                        if "inner_index" in each:
                            if each["inner_index"] == p:
                                element['group_id'] = each['group_id']

                else:
                    group_id = "group_" + str(uuid.uuid4())
                    element['group_id'] = group_id
            else:
                relation_id = "relation_" + str(uuid.uuid4())
                element['relation_id'] = relation_id
    write_jsonl_file(f"Data/{filename}/{filename}_EX_pro.jsonl", ex_list)



def check_group_id(inner_index, merge_list):
    for item in merge_list:
        if item[1]["inner_index"] == inner_index:
            return item[0]["inner_index"]
    return 0



if __name__ == '__main__':
    print("开始执行step2-2 结点融合")
    filename = "gmzz_1"
    entities_fusion(filename)