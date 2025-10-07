import json
import re

from openai import OpenAI

from Neo4j_tools import create_chunk_node, create_entity_node, create_relationship_to_chunk, \
    create_relationships_to_entity, create_groups, fetch_all_nodes, fetch_nodes
from llm_core import call_LLM
from logger_config import setup_logger
from tools import read_jsonl_to_list, count_txt_files, write_jsonl_file, append_dict_to_jsonl, count_lines, \
    reset_jsonl_file

from concurrent.futures import ThreadPoolExecutor, as_completed
# 日志文件路径
log_file_path = 'UniRAG.log'
# 设置日志记录器
logger = setup_logger(log_file_path)


MERGE_PROMPT = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given several entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context. 
Your output language needs to be consistent with the language of the descriptions.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


def single_merge(filename: str, LLM_Type: str, model_name: str, group: list):
    # group为0则跳过
    if len(group) == 0:
        logger.info("group 为空，跳过")
        return


    entity_name = []
    description_list = []
    source_index = []
    for entity in group:
        # 如果名称没有出现过则加入列表
        name = entity["n"]["name"]
        if name not in entity_name:
            entity_name.append(name)
        description_list.append({"description": entity["n"]["description"], "index": entity["n"]["source_index"]})
        source_index.append(entity["n"]["source_index"])

    # 把description_list 按照index排序
    description_list.sort(key=lambda x: x["index"])
    description_input = ""
    for description in description_list:
        description_input += f"{description['description']}————From part {description['index']} of the original article\n"

    # print("名称：")
    # print(entity_name)
    #
    # print("描述：")
    # print(description_input)

    processed_group = read_jsonl_to_list(f"Data/{filename}/{filename}_group_fusion.jsonl")
    # 如果列表里的group_info里存在本group里的group[0]["n"]["group_id"] 则直接跳过
    for group_info in processed_group:
        if group_info["group_id"] == group[0]["n"]["group_id"]:
            if "group_name_list" not in group_info:
                group_info["group_name_list"] = entity_name
            if "source_index" not in group_info:
                group_info["source_index"] = source_index

            # 检查一下格式是否正确

            logger.info("group_id 重复，跳过")
            return

    prompt = MERGE_PROMPT.format(entity_name=str(entity_name), description_list=description_input)

    ans = call_LLM(prompt=prompt, LLM_type=LLM_Type, model_name=model_name)
    group_info = {"group_id": group[0]["n"]["group_id"], "group_name": entity_name[0],
                  "description": ans, "group_name_list": entity_name, "source_index": source_index}
    append_dict_to_jsonl(f"Data/{filename}/{filename}_group_fusion.jsonl", group_info)
    logger.info(f"{entity_name[0]}的局部融合结果已经生成")


def local_merge(filename: str, LLM_Type: str, model_name: str, workers):
    groups = fetch_all_nodes("Group")
    stay_groups = []
    to_do_groups = []
    for group in groups:
        group_id = group["n"]["group_id"]
        group = fetch_nodes("Entity","group_id", group_id)
        if len(group) > 1:
            to_do_groups.append({"group_id": group_id, "group": group})
        else:
            stay_groups.append({"group_id": group_id, "group": group})

    # 先处理那些只有一个结点的Group
    reset_jsonl_file(f"Data/{filename}/{filename}_group_stay.jsonl")
    for each in stay_groups:
        group_id = each["group_id"]
        group = fetch_nodes("Entity","group_id", group_id)
        group_info = {"group_id": group_id, "group_name": group[0]["n"]["name"],
                      "description": group[0]["n"]["description"], "group_name_list": [group[0]["n"]["name"]],
                      "source_index": [group[0]["n"]["source_index"]]}
        append_dict_to_jsonl(f"Data/{filename}/{filename}_group_stay.jsonl", group_info)

    write_jsonl_file(f"Data/{filename}/{filename}_group_fusion_temp.jsonl", to_do_groups)
    logger.info(f"总共有{len(groups)}个组，需要处理的有{len(to_do_groups)}个组")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for group in to_do_groups:
            # 提交任务到线程池
            future = executor.submit(single_merge, filename, LLM_Type, model_name, group["group"])
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()  # 获取任务的返回结果
            except Exception as e:
                print(f"任务执行出错: {e}")
    logger.info("处理完了")


if __name__ == '__main__':

    filename = "gmzz_1"
    LLM_type = "Silicon"
    model_name = "Pro/deepseek-ai/DeepSeek-V3"
    workers = 10

    print("开始执行 step3-1-局部融合.py")
    # local_merge("ydt", "Silicon", "Pro/deepseek-ai/DeepSeek-V3")
    groups = fetch_all_nodes("Group")
    to_do_groups = []
    for group in groups:
        group_id = group["n"]["group_id"]
        group = fetch_nodes("Entity", "group_id", group_id)
        if len(group) > 1:
            to_do_groups.append({"group_id": group_id, "group": group})

    write_jsonl_file(f"Data/{filename}/{filename}_group_fusion_temp.jsonl", to_do_groups)

    total_count = len(to_do_groups)

    reset_jsonl_file(f"Data/{filename}/{filename}_group_fusion.jsonl")
    reset_jsonl_file(f"Data/{filename}/{filename}_group_stay.jsonl")

    processed_count = count_lines(f"Data/{filename}/{filename}_group_fusion.jsonl")
    while processed_count < total_count:
        local_merge(filename, LLM_type, model_name, workers)
        processed_count = count_lines(f"Data/{filename}/{filename}_group_fusion.jsonl")

    # local_merge("ydt", "DeepSeek", "deepseek-chat", workers)