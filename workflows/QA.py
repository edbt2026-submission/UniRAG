import json
import re

import numpy as np

from llm_core import call_LLM, get_embedding, rerank
from logger_config import setup_logger
from tools import read_jsonl_to_list, count_txt_files, extract_json, search_chunks, truncate_list_by_token_size, \
    search_group, search_global, search_relation, count_list_tokens, truncate_list_by_token_size2, append_dict_to_jsonl

from concurrent.futures import ThreadPoolExecutor, as_completed
# 日志文件路径
log_file_path = 'UniRAG.log'
# 设置日志记录器
logger = setup_logger(log_file_path)


PROMPT_KEYWORD= """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.
Your output language needs to be consistent with the language of the query.

---Instructions---

- Your output language needs to be consistent with the language of the query.
- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPT_NAIVE = """---Role---

You are a helpful assistant responding to questions about documents provided.

---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

Multiple Paragraphs

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
Your output language needs to be consistent with the language of the query.

{query}
"""


PROMPT_GLOBAL = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

"Multiple Paragraphs"

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
Your output language needs to be consistent with the language of the query.

{query}
"""

import asyncio

async def QA(filename, query, query_mode, LLM_type, model_name, k):

    if query_mode == "naive":

        v = get_embedding(text=query, Embedding_type="Silicon", model_name="Pro/BAAI/bge-m3")
        related_chunks = search_chunks(filename, v, 60)
        related_chunks = truncate_list_by_token_size(related_chunks, 4000)
        section = "--New Chunks--\n".join([c["text"] for c in related_chunks])
        prompt = PROMPT_NAIVE.format(content_data=section, query=query)
        logger.info(f"准备访问LLM")
        ans = call_LLM(prompt=prompt, LLM_type=LLM_type, model_name=model_name)
        print(ans)
        return ans

    elif query_mode == "keyword":
        ans = call_LLM(prompt=PROMPT_KEYWORD.format(query=query), LLM_type=LLM_type, model_name=model_name)
        ans = extract_json(ans)

        logger.info(f"keywords提取结果：{ans}")
        keywords = json.loads(ans)

        context_pool = []

        # 处理highlevel
        if len(keywords["high_level_keywords"]) > 0:
            highlevelwords = " ".join(word for word in keywords["high_level_keywords"])
            v_h = get_embedding(text=highlevelwords, Embedding_type="Silicon", model_name="Pro/BAAI/bge-m3")
            # highlevelwords匹配
            highlevelwords_global = search_global(filename, v_h, k)
            logger.info("highlevelwords匹配的前k个global：" + str(highlevelwords_global))
            for each in highlevelwords_global:
                text = "\"" + each["global_name"] + "\": " + each["description"] + "\n\n"
                context_pool.append(text)

            highlevelwords_group = search_group(filename, v_h, k)
            logger.info("highlevelwords匹配的前k个group：" + str(highlevelwords_group))
            for each in highlevelwords_group:
                text = "\"" + each["group_name"] + "\": " + each["description"] + "\n\n"
                context_pool.append(text)

            highlevelwords_relation = search_relation(filename, v_h, k)
            logger.info("highlevelwords匹配的前k个relation：" + str(highlevelwords_relation))
            for each in highlevelwords_relation:
                text = "relationship: " + each["description"]
                context_pool.append(text)


        # 处理lowlevel
        if len(keywords["low_level_keywords"]) > 0:
            lowlevelwords = " ".join(word for word in keywords["low_level_keywords"])

            v_l = get_embedding(text=lowlevelwords, Embedding_type="Silicon", model_name="Pro/BAAI/bge-m3")
            # lowlevelwords匹配
            lowlevelwords_global = search_global(filename, v_l, k)
            logger.info("lowlevelwords匹配的前k个global：" + str(lowlevelwords_global))
            for each in lowlevelwords_global:
                text = "\"" + each["global_name"] + "\": " + each["description"] + "\n\n"
                context_pool.append(text)

            lowlevelwords_group = search_group(filename, v_l, k)
            logger.info("lowlevelwords匹配的前k个group：" + str(lowlevelwords_group))
            for each in lowlevelwords_group:
                text = "\"" + each["group_name"] + "\": " + each["description"] + "\n\n"
                context_pool.append(text)

            lowlevelwords_relation = search_relation(filename, v_l, k)
            logger.info("lowlevelwords匹配的前k个relation：" + str(lowlevelwords_relation))
            for each in lowlevelwords_relation:
                text = "\"relationship\": " + each["description"] + "\n\n"
                context_pool.append(text)

        v = get_embedding(text=query, Embedding_type="Silicon", model_name="Pro/BAAI/bge-m3")



        # query直接匹配
        query_global = search_global(filename, v, k)
        logger.info("问题直接匹配的前k个global：" + str(query_global))
        for each in query_global:
            text = "\""+each["global_name"]+"\": " + each["description"] + "\n\n"
            context_pool.append(text)
        query_group = search_group(filename, v, k)
        logger.info("问题直接匹配的前k个group：" + str(query_group))
        for each in query_group:
            text = "\""+each["group_name"]+"\": " + each["description"] + "\n\n"
            context_pool.append(text)
        query_relation = search_relation(filename, v, k)
        logger.info("问题直接匹配的前k个relation：" + str(query_relation))
        for each in query_relation:
            text = "\"relationship\": " + each["description"] + "\n\n"
            context_pool.append(text)

        print("context_pool去重前："+str(len(context_pool)))
        context_pool = list(set(context_pool))
        print("context_pool去重后："+str(len(context_pool)))

        reranked = rerank(query, context_pool, top_n=k)

        final_context = []
        for each in reranked['results']:
            final_context.append(context_pool[each['index']])

        print("原始长度：" + str(count_list_tokens(final_context)))
        final_context = truncate_list_by_token_size2(final_context, 4000)
        print("截断后长度：" + str(count_list_tokens(final_context)))

        ans = call_LLM(prompt=PROMPT_GLOBAL.format(context_data="\n".join(final_context), query=query), LLM_type=LLM_type, model_name=model_name)
        print(ans)
        return ans
    else:
        return None


async def main(questions, filename, LLM_type, model_name_1, model_name_2, k):

    for question in questions:
        naive_ans_1, global_ans_1, naive_ans_2, global_ans_2 = await asyncio.gather(
            QA(filename, question, "naive", LLM_type, model_name_1, k),
            QA(filename, question, "keyword", LLM_type, model_name_1, k),
            QA(filename, question, "naive", LLM_type, model_name_2, k),
            QA(filename, question, "keyword", LLM_type, model_name_2, k-20)
        )

        high_level_models = {"朴素RAG回答": naive_ans_1, "UniRAG回答": global_ans_1}
        low_level_models = {"朴素RAG回答": naive_ans_2, "UniRAG回答": global_ans_2}

        # 做记录
        entry = {"query": question, "DeepSeek-V3(性能)": high_level_models, "Qwen2.5-7B(效率)": low_level_models}
        append_dict_to_jsonl("TEST/QA_test.jsonl", entry)

if __name__ == '__main__':
    filename = "gmzz_1"

    LLM_type = "Silicon"
    model_name_1 = "Pro/deepseek-ai/DeepSeek-V3"
    model_name_2 = "Pro/Qwen/Qwen2.5-7B-Instruct" # 你也可以换成其他的试试 比如14B 32B 72B 72B几乎是一般个人部署的上限了

    questions = ["这篇小说在你看来目前的世界观是怎么样的？", "你可以和我详细说说克莱恩使用神秘学知识来解决的案件的详细过程吗？"]
    asyncio.run(main(questions, filename, LLM_type, model_name_1, model_name_2, 60))







