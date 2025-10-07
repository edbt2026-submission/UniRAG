import json
import re

from openai import OpenAI

from Neo4j_tools import fetch_all_nodes, add_embedding_attr, add_embedding_attr_global, get_relations, add_relation_attr
from llm_core import call_LLM, get_embedding
from logger_config import setup_logger
from tools import read_jsonl_to_list, count_txt_files, reset_jsonl_file, append_dict_to_jsonl, count_lines, \
    write_jsonl_file

from concurrent.futures import ThreadPoolExecutor, as_completed
# 日志文件路径
log_file_path = 'UniRAG.log'
# 设置日志记录器
logger = setup_logger(log_file_path)


def group_embedding(filename, Embedding_type, model_name, group, processed_list):
    group_id = group["n"]["group_id"]
    group_name = group["n"]["group_name"]
    for each in processed_list:
        if each["group_id"] == group_id:
            logger.info(f"{group_name} 已经处理过, 跳过")
            return

    group_name = group["n"]["group_name"]
    logger.info(f"正在处理组 {group_name}")
    name_embedding = get_embedding(text=group_name, Embedding_type=Embedding_type, model_name=model_name)
    group_description = group["n"]["description"]
    description_embedding = get_embedding(text=group_description, Embedding_type=Embedding_type, model_name=model_name)
    hybrid = group_name + ": " + group_description
    hybrid_embedding = get_embedding(text=hybrid, Embedding_type=Embedding_type, model_name=model_name)
    group_id = group["n"]["group_id"]
    add_embedding_attr(label="Group", key="group_id", value=group_id, name=name_embedding, desc=description_embedding,
                       hybrid=hybrid_embedding)
    entry = {"group_id": group_id, "group_name": group_name, "description": group_description,
             "name_embedding": name_embedding, "description_embedding": description_embedding,
             "hybrid_embedding": hybrid_embedding}
    append_dict_to_jsonl(f"Data/{filename}/{filename}_group_embedding.jsonl", entry)


def global_embedding(filename, Embedding_type, model_name, g, processed_list):
    global_name = g["n"]["global_name"]
    global_id = g["n"]["global_id"]

    for each in processed_list:
        if each["global_id"] == global_id:
            logger.info(f"{global_name} 已经处理过, 跳过")
            return



    logger.info(f"正在处理全局 {global_name}")
    global_description = g["n"]["description"]
    description_embedding = get_embedding(text=global_description, Embedding_type=Embedding_type, model_name=model_name)
    add_embedding_attr_global(label="Global", key="global_id", value=global_id, desc=description_embedding)
    entry = {"global_id": global_id, "global_name": global_name, "description": global_description,
             "description_embedding": description_embedding}
    append_dict_to_jsonl(f"Data/{filename}/{filename}_global_embedding.jsonl", entry)


def chunk_embedding(filename, Embedding_type, model_name, chunk, processed_list):
    chunk_id = chunk["n"]["id"]
    index = chunk["n"]["index"]

    for each in processed_list:
        if each["id"] == chunk_id:
            logger.info(f"{chunk_id} 已经处理过, 跳过")
            return

    logger.info(f"正在处理原文段 {chunk_id}")
    chunk_text = chunk["n"]["text"]
    chunk_embedding = get_embedding(text=chunk_text, Embedding_type=Embedding_type, model_name=model_name)
    add_embedding_attr_global(label="Chunk", key="id", value=chunk_id, desc=chunk_embedding)
    entry = {"index": index, "id": chunk_id, "text": chunk_text, "embedding": chunk_embedding}
    append_dict_to_jsonl(f"Data/{filename}/{filename}_chunk_embedding.jsonl", entry)


def relation_embedding(filename, Embedding_type, model_name, relation, processed_list):
    relation_id = relation["id"]

    for each in processed_list:
        if each["id"] == relation_id:
            logger.info(f"{relation_id} 已经处理过, 跳过")
            return

    logger.info(f"正在处理关系 {relation_id}")
    desc = relation["description"]
    relation_embedding = get_embedding(text=desc, Embedding_type=Embedding_type, model_name=model_name)
    add_relation_attr(id=relation_id, k="embedding", v=relation_embedding)
    entry = {"id": relation_id, "description": desc, "embedding": relation_embedding}
    append_dict_to_jsonl(f"Data/{filename}/{filename}_relations_embedding.jsonl", entry)


def groups_embedding(filename, Embedding_type, model_name, workers):
    groups_list = fetch_all_nodes("Group")
    processed_list = read_jsonl_to_list(f"Data/{filename}/{filename}_group_embedding.jsonl")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for group in groups_list:
            future = executor.submit(group_embedding, filename, Embedding_type, model_name, group, processed_list)
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()  # 获取任务的返回结果
            except Exception as e:
                print(f"任务执行出错: {e}")


def chunks_embedding(filename, Embedding_type, model_name, workers):
    chunks_list = fetch_all_nodes("Chunk")
    processed_list = read_jsonl_to_list(f"Data/{filename}/{filename}_chunk_embedding.jsonl")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for chunk in chunks_list:
            future = executor.submit(chunk_embedding, filename, Embedding_type, model_name, chunk, processed_list)
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()  # 获取任务的返回结果
            except Exception as e:
                print(f"任务执行出错: {e}")


def globals_embedding(filename, Embedding_type, model_name, workers):
    globals_list = fetch_all_nodes("Global")
    processed_list = read_jsonl_to_list(f"Data/{filename}/{filename}_global_embedding.jsonl")


    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for g in globals_list:
            future = executor.submit(global_embedding, filename, Embedding_type, model_name, g, processed_list)
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()  # 获取任务的返回结果
            except Exception as e:
                print(f"任务执行出错: {e}")


def relations_embedding(filename, Embedding_type, model_name, workers):
    relations = get_relations()
    relations = [tuple(rel.items()) for rel in relations]
    relations = set(relations)
    relations = [dict(rel) for rel in relations]
    # print(relations)
    write_jsonl_file(f"Data/{filename}/{filename}_relations_before_embedding.jsonl", relations)
    processed_list = read_jsonl_to_list(f"Data/{filename}/{filename}_relations_embedding.jsonl")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for relation in relations:
            future = executor.submit(relation_embedding, filename, Embedding_type, model_name, relation, processed_list)
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()  # 获取任务的返回结果
            except Exception as e:
                print(f"任务执行出错: {e}")




if __name__ == '__main__':
    print("开始执行 step-4-1-Embedding.py")
    # groups_embedding("ydt", "Silicon", "Pro/BAAI/bge-m3", 10)

    filename = "gmzz_1"
    workers = 10

    # 第一次执行记得解除下面的注释
    reset_jsonl_file(f"Data/{filename}/{filename}_global_embedding.jsonl")
    reset_jsonl_file(f"Data/{filename}/{filename}_group_embedding.jsonl")
    reset_jsonl_file(f"Data/{filename}/{filename}_chunk_embedding.jsonl")
    reset_jsonl_file(f"Data/{filename}/{filename}_relations_embedding.jsonl")


    total_counts = count_lines(f"Data/{filename}/{filename}_group_all.jsonl")
    processed_counts = count_lines(f"Data/{filename}/{filename}_group_embedding.jsonl")

    while processed_counts < total_counts:
        logger.info(f"正在处理组 {processed_counts}/{total_counts}")
        groups_embedding(filename, "Silicon", "Pro/BAAI/bge-m3", workers)
        processed_counts = count_lines(f"Data/{filename}/{filename}_group_embedding.jsonl")


    total_counts = count_lines(f"Data/{filename}/{filename}_group_after_global.jsonl")
    processed_counts = count_lines(f"Data/{filename}/{filename}_global_embedding.jsonl")


    while processed_counts < total_counts:
        logger.info(f"正在处理全局 {processed_counts}/{total_counts}")
        globals_embedding(filename, "Silicon", "Pro/BAAI/bge-m3", workers)
        processed_counts = count_lines(f"Data/{filename}/{filename}_global_embedding.jsonl")

    total_counts = count_txt_files(f"Data/Chunks/{filename}_Small")
    logger.info(f"原文段总数：{total_counts}")

    processed_counts = count_lines(f"Data/{filename}/{filename}_chunk_embedding.jsonl")
    logger.info(f"已处理原文段数：{processed_counts}")

    while processed_counts < total_counts:
        chunks_embedding(filename, "Silicon", "Pro/BAAI/bge-m3", workers)
        processed_counts = count_lines(f"Data/{filename}/{filename}_chunk_embedding.jsonl")

    relations_embedding(filename, "Silicon", "Pro/BAAI/bge-m3", workers)

    total_counts = count_lines(f"Data/{filename}/{filename}_relations_before_embedding.jsonl")
    processed_counts = count_lines(f"Data/{filename}/{filename}_relations_embedding.jsonl")
    while processed_counts < total_counts or total_counts == 0:
        logger.info(f"正在处理关系 {processed_counts}/{total_counts}")
        relations_embedding(filename, "Silicon", "Pro/BAAI/bge-m3", workers)
        processed_counts = count_lines(f"Data/{filename}/{filename}_relations_embedding.jsonl")