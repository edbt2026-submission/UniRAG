import json
import re

import faiss
import numpy as np
from openai import OpenAI

from Neo4j_tools import fetch_all_nodes, add_embedding_attr
from llm_core import call_LLM, get_embedding
from logger_config import setup_logger
from tools import read_jsonl_to_list, count_txt_files, reset_jsonl_file, append_dict_to_jsonl, search_similar_group

from concurrent.futures import ThreadPoolExecutor, as_completed
# 日志文件路径
log_file_path = 'UniRAG.log'
# 设置日志记录器
logger = setup_logger(log_file_path)


def construct_faiss_index(filename):
    vectors_data = read_jsonl_to_list(f"Data/{filename}/{filename}_group_embedding.jsonl")

    # 提取 name_embedding 数据
    name_vectors = np.array([item["name_embedding"] for item in vectors_data], dtype=np.float32)
    d = name_vectors.shape[1]
    name_index = faiss.IndexFlatL2(d)
    name_index.add(name_vectors)
    faiss.write_index(name_index, f"Vector/{filename}_name_index.faiss")

    # 提取 description_embedding 数据
    description_vectors = np.array([item["description_embedding"] for item in vectors_data], dtype=np.float32)
    d = description_vectors.shape[1]
    description_index = faiss.IndexFlatL2(d)
    description_index.add(description_vectors)
    faiss.write_index(description_index, f"Vector/{filename}_description_index.faiss")

    # 提取 hybrid_embedding 数据
    hybrid_vectors = np.array([item["hybrid_embedding"] for item in vectors_data], dtype=np.float32)
    d = hybrid_vectors.shape[1]
    hybrid_index = faiss.IndexFlatL2(d)
    hybrid_index.add(hybrid_vectors)
    faiss.write_index(hybrid_index, f"Vector/{filename}_hybrid_index.faiss")

    vectors_data = read_jsonl_to_list(f"Data/{filename}/{filename}_global_embedding.jsonl")
    description_vectors = np.array([item["description_embedding"] for item in vectors_data], dtype=np.float32)
    d = description_vectors.shape[1]
    global_index = faiss.IndexFlatL2(d)
    global_index.add(description_vectors)
    faiss.write_index(global_index, f"Vector/{filename}_global_index.faiss")

    vectors_data = read_jsonl_to_list(f"Data/{filename}/{filename}_chunk_embedding.jsonl")
    description_vectors = np.array([item["embedding"] for item in vectors_data], dtype=np.float32)
    d = description_vectors.shape[1]
    chunk_index = faiss.IndexFlatL2(d)
    chunk_index.add(description_vectors)
    faiss.write_index(chunk_index, f"Vector/{filename}_chunk_index.faiss")

    vectors_data = read_jsonl_to_list(f"Data/{filename}/{filename}_relations_embedding.jsonl")
    description_vectors = np.array([item["embedding"] for item in vectors_data], dtype=np.float32)
    d = description_vectors.shape[1]
    relation_index = faiss.IndexFlatL2(d)
    relation_index.add(description_vectors)
    faiss.write_index(relation_index, f"Vector/{filename}_relation_index.faiss")



if __name__ == '__main__':
    filename = "gmzz_1"

    construct_faiss_index(filename)

    # text = "下鸭矢三郎"
    #
    # v = get_embedding(text=text, Embedding_type="Silicon", model_name="Pro/BAAI/bge-m3")
    # vector = np.array(v, dtype=np.float32)
    #
    # search_similar_group("ydt", vector, "name", 30)