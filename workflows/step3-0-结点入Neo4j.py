import json
import re
import uuid

from openai import OpenAI

from Neo4j_tools import create_chunk_node, create_entity_node, create_relationship_to_chunk, \
    create_relationships_to_entity, create_groups
from llm_core import call_LLM
from logger_config import setup_logger
from tools import read_jsonl_to_list, count_txt_files, read_text_file, read_chunk, append_dict_to_jsonl

from concurrent.futures import ThreadPoolExecutor, as_completed
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
        sorted_json_list = sorted(json_list, key=lambda x: (x['index']))

    # 将排序后的JSON对象写入新的JSONL文件
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for json_obj in sorted_json_list:
            # 将JSON对象转换为字符串并写入文件，每个对象占一行
            file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')


def create_chunks_nodes(filename):
    ex_list = read_jsonl_to_list(f"Data/{filename}/{filename}_EX.jsonl")  # 读取 JSONL 文件
    for ex in ex_list:
        chunk_uuid = "chunk_" + str(uuid.uuid4())  # 生成 UUID
        text = read_chunk(filename, ex['index_L'], ex['index_S'])  # 读取 chunk 文本
        index = ex['index']
        create_chunk_node(chunk_uuid, filename, text, index)


def create_entities_nodes(filename):
    ex_list = read_jsonl_to_list(f"Data/{filename}/{filename}_EX_pro.jsonl")
    for ex in ex_list:
        entity_id_map = {}
        entities_list = ex['EX']
        index  = ex['index']
        for entry in entities_list:
            if 'entity' in entry:
                entity = entry["entity"]
                entity_id = entry["entity_id"]  # 生成UUID
                group_id = entry["group_id"]
                entity["id"] = entity_id  # 将ID添加到entity对象
                entity_id_map[entity["name"]] = entity_id  # 将实体名称和ID映射存储
                name = entry['entity']['name']
                type = entry['entity']['type']
                description = entry['entity']['description']
                create_entity_node(entity_id, name, type, description, index, group_id)
                create_relationship_to_chunk(entity_id, index)
        for entry in entities_list:
            if 'relationship' in entry:
                relationship = entry["relationship"]
                source_name = relationship["source"]
                target_name = relationship["target"]
                relation_id = entry["relation_id"]
                relationship["source_id"] = entity_id_map.get(source_name)  # 根据名称获取source_id
                relationship["target_id"] = entity_id_map.get(target_name)  # 根据名称获取target_id
                relationship["id"] = relation_id
                create_relationships_to_entity(relationship["source_id"], relationship["target_id"],
                                              relationship["keywords"], relationship["description"],
                                              relationship["strength"], relation_id)
        # append_dict_to_jsonl(f"Data/{filename}/{filename}_EX_Pro.jsonl", ex)


def create_UUID(filename):
    ex_list = read_jsonl_to_list(f"Data/{filename}/{filename}_EX.jsonl")


if __name__ == '__main__':
    filename = "gmzz_1"
    create_chunks_nodes(filename)
    create_entities_nodes(filename)
    create_groups()