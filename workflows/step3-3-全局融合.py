import json
import re
import uuid
from collections import defaultdict

from openai import OpenAI

from Neo4j_tools import create_chunk_node, create_entity_node, create_relationship_to_chunk, \
    create_relationships_to_entity, create_groups, fetch_all_nodes, fetch_nodes
from llm_core import call_LLM
from logger_config import setup_logger
from tools import read_jsonl_to_list, count_txt_files, write_jsonl_file, append_dict_to_jsonl, count_lines, \
    reset_jsonl_file, has_common_element

from concurrent.futures import ThreadPoolExecutor, as_completed
# 日志文件路径
log_file_path = 'UniRAG.log'
# 设置日志记录器
logger = setup_logger(log_file_path)


MERGE_PROMPT = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given several entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
Please provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context. 
Your output language needs to be consistent with the language of the descriptions.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


def singel_merge(filename: str, LLM_Type: str, model_name: str, groups: list, merged_id):
    if groups[0]["group_id"] in merged_id:
        name = groups[0]["group_name"]
        logger.info(f"{name}已经融合过了，跳过")
        return



    elif len(groups) == 1:
        entry = {"global_id": "global_"+str(uuid.uuid4()), "global_name": groups[0]["group_name"],
                 "global_description": groups[0]["description"], "source_index": groups[0]["source_index"],
                 "group_ids":[groups[0]["group_id"]], "other_names": [groups[0]["group_name"]]}
        append_dict_to_jsonl(f"Data/{filename}/{filename}_group_after_global.jsonl", entry)
    else:
        entity_name = []
        description_list = ""
        for group in groups:
            entity_name += group["group_name_list"]
            description_list += group["description"] + "————From part " + str(group["source_index"]) + "of the original article \n"

        entity_name = list(set(entity_name))

        ans = call_LLM(prompt=MERGE_PROMPT.format(entity_name=str(entity_name), description_list=str(description_list)), LLM_type=LLM_Type, model_name=model_name, json_output=False)
        entry = {"global_id": "global_"+str(uuid.uuid4()), "global_name": groups[0]["group_name"],
                 "global_description": ans, "source_index": groups[0]["source_index"],
                 "group_ids": [group["group_id"] for group in groups],
                 "other_names": entity_name}
        append_dict_to_jsonl(f"Data/{filename}/{filename}_group_after_global.jsonl", entry)
        logger.info(f"{groups[0]['group_name']}的全局融合结果已经生成")
        merged_id.append(groups[0]["group_id"])


def union(filename: str):
    fusion = read_jsonl_to_list(f"Data/{filename}/{filename}_group_fusion.jsonl")
    stay = read_jsonl_to_list(f"Data/{filename}/{filename}_group_stay.jsonl")
    all = fusion + stay
    write_jsonl_file(f"Data/{filename}/{filename}_group_all.jsonl", all)
    groups = read_jsonl_to_list(f"Data/{filename}/{filename}_group_all.jsonl")

    parent = {}
    rank = {}


    def find(x):
        if x not in parent:
            parent[x] = x
            rank[x] = 1
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        x_root = find(x)
        y_root = find(y)
        if rank[x_root] < rank[y_root]:
            parent[x_root] = y_root
        else:
            parent[y_root] = x_root
            if rank[x_root] == rank[y_root]:
                rank[y_root] += 1

    for group in groups:
        names = group["group_name_list"]
        pivot = names[0]
        for s in names[1:]:
            union(pivot, s)

    connected_groups = defaultdict(list)
    for group in groups:
        names = group["group_name_list"]
        root = find(names[0])
        connected_groups[root].append(group)

    final = list(connected_groups.values())
    write_jsonl_file(f"Data/{filename}/{filename}_group_before_global.jsonl", final)



def global_merge(filename: str, LLM_Type: str, model_name: str, workers) -> None:

    final = read_jsonl_to_list(f"Data/{filename}/{filename}_group_before_global.jsonl")


    merged_id = []
    merged_all = read_jsonl_to_list(f"Data/{filename}/{filename}_group_after_global.jsonl")
    for merged in merged_all:
        merged_id.append(merged["group_ids"][0])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for group in final:
            future = executor.submit(singel_merge, filename, LLM_type, model_name, group, merged_id)
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()  # 获取任务的返回结果
            except Exception as e:
                print(f"任务执行出错: {e}")
    logger.info("全局融合结束")





if __name__ == '__main__':

    filename = "gmzz_1"
    LLM_type = "Silicon"
    model_name = "Pro/deepseek-ai/DeepSeek-V3"
    workers = 20

    print("开始执行 step3-3-全局融合.py")

    reset_jsonl_file(f"Data/{filename}/{filename}_group_before_global.jsonl")
    union(filename)

    reset_jsonl_file(f"Data/{filename}/{filename}_group_after_global.jsonl")
    total_count = count_lines(f"Data/{filename}/{filename}_group_before_global.jsonl")
    processed_count = count_lines(f"Data/{filename}/{filename}_group_after_global.jsonl")
    while processed_count < total_count:
        logger.info(f"{processed_count}/{total_count}")
        global_merge(filename, LLM_type, model_name, workers)
        processed_count = count_lines(f"Data/{filename}/{filename}_group_after_global.jsonl")

