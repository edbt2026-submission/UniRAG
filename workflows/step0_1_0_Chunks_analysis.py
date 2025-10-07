import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tools import read_jsonl_to_list, append_dict_to_jsonl, count_files, extract_json, reset_jsonl_file, count_lines
from llm_core import call_LLM
from logger_config import setup_logger

# 日志文件路径
log_file_path = 'UniRAG.log'
# 设置日志记录器
logger = setup_logger(log_file_path)

PROMPT_TEMPLATE_1 = """{text}

Based on the above content, I need you to complete the following tasks:
1. Briefly describe what type of text this is, such as "novel," "thesis," "tutorial," "proposal," etc.
2. The above content is the first part of a lengthy original document. I need you to summarize the text so that readers can smoothly continue reading from the second part.
3. I now want to construct a knowledge graph for this part. Do you have any suggestions?
# The suggestions for knowledge graph construction should only consider entities, relationships, and attributes. There is no need to provide tool-related suggestions.

4. Your output language needs to be consistent with the language of the content.

Your response should be in JSON format, with the specific structure as follows:
{{
"Text_Type": <The type of text you believe it is>,
"Summary": <Your summary>,
"Suggestions": <Your suggestions for knowledge graph construction, In natural language string format>
}}

Your JSON:"""

PROMPT_TEMPLATE_2 = """First, please review the previous summary: {summary}. Next is the {i}th part of a document:\n{text}\nBased on the above content, I need you to complete the following tasks:
1. I need you to summarize the text so that readers can smoothly continue reading from the next part.
2. I now want to construct a knowledge graph for this part. Do you have any suggestions? For the previous part, I received the following advice: {advice}. Based on the content of this part, please generate a new suggestion.

# The suggestions for knowledge graph construction should only consider entities, relationships, and attributes. There is no need to provide tool-related suggestions.
3. Your output language needs to be consistent with the language of the content.

Your response should be in JSON format, with the specific structure as follows:
{{
"Summary": <Your summary>,
"Suggestions": <Your suggestions for knowledge graph construction>
}}

Your JSON:"""

PROMPT_TEMPLATE_3 = '''### Objective ###
Given a summary of a text document, suggestions for knowledge graph construction, and a specific small portion of the original text, identify all entities of these types from the text and all relationships between the identified entities.

### Steps ###
1. Refer to the document summary and identify all entities in the specific small portion of the original text. For each identified entity, extract the following information:
   - entity_name: The name of the entity. If it is in English, use uppercase.
   - entity_type: Such as person, character, technology, organization, event, location, concept, etc. You can freely interpret this.
   - entity_description: A comprehensive description of the entity's attributes and activities.
   Format each entity as JSON: {{"entity":{{"name":<entity_name>,"type":<entity_type>,"description":<entity_description>}}}}
   Note: In this step, you cannot extract entities from the document summary. The entities you extract must come from the small portion of the original text.

2. From the entities identified in Step 1, determine all pairs (source_entity, target_entity) that are "clearly related."
   For each related pair, extract the following information:
   - source_entity: The name of the source entity, consistent with Step 1.
   - target_entity: The name of the target entity, consistent with Step 1.
   - relationship_description: Explain why you think the source entity and target entity are related.
   - relationship_strength: A score indicating the strength of the relationship between entities (range 1-10).
   - relationship_keywords: One or more high-level keywords summarizing the overall nature of the relationship, listed as a list, focusing on concepts or themes rather than specific details.
   Format each relationship as JSON: {{"relationship":{{"source":<source_entity>,"target":<target_entity>,"description":<relationship_description>,"keywords":[<relationship_keywords>],"strength":<relationship_strength>}}}}

3. Identify high-level keywords that summarize the main concepts, themes, or topics of the entire article. These should capture the overall ideas presented in the document. List them in a list.
   Format the content-level keywords as JSON: {{"keywords":[<high_level_keywords>]}}

4. Return the output as a single list of all entities and relationships identified in Steps 1 and 2. The returned result should be integrated into a single JSON list.

5. Your output language needs to be consistent with the language of the content. Make sure the entities' name, type, description are consistent with the language of the given content.

### Text Data ###
######################
Summary: {summary}
Knowledge Graph Construction Suggestions: {advice}
A small portion of the original text: {input_text}

Your output JSON format is as follows:
[
    {{"entity":{{"name":<entity_name>,"type":<entity_type>,"description":<entity_description>}}}},
    {{"entity":{{"name":<entity_name>,"type":<entity_type>,"description":<entity_description>}}}},
    ······
    {{"relationship":{{"source":<source_entity>,"target":<target_entity>,"description":<relationship_description>,"keywords":[<relationship_keywords>],"strength":<relationship_strength>}}}},
    {{"relationship":{{"source":<source_entity>,"target":<target_entity>,"description":<relationship_description>,"keywords":[<relationship_keywords>],"strength":<relationship_strength>}}}},
    ······
    {{"keywords":[<high_level_keywords>]}}
]

Your final output JSON:
'''


# 读取JSONL文件，并按照index_L和index_S排序
def sort_jsonl_file(input_file_path, output_file_path):
    # 读取文件并解析每一行为JSON对象
    with open(input_file_path, 'r', encoding='utf-8') as file:
        # 使用生成器表达式来加载每一行的JSON
        json_list = (json.loads(line) for line in file)

        # 按照index_L和index_S排序
        sorted_json_list = sorted(json_list, key=lambda x: (x['index_L'], x['index_S']))

    # 将排序后的JSON对象写入新的JSONL文件
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for json_obj in sorted_json_list:
            # 将JSON对象转换为字符串并写入文件，每个对象占一行
            file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')


def count_txt_files(folder_path):
    # 初始化计数器
    count = 0

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是txt文件
        if filename.endswith(".txt"):
            count += 1

    return count


def analyze_text_large(textname, LLM_type, model_name, json_mode=True):
    total_index = count_txt_files(f'Data/Chunks/{textname}_Large')
    with open(f"Data/Chunks/{textname}_Large/chunk_001.txt", "r", encoding="utf-8") as f:
        text1 = f.read()
    input_text1 = PROMPT_TEMPLATE_1.format(text=text1)
    logger.info(f"{textname} Large Text Analysis Started")
    if json_mode:
        ans1 = call_LLM(prompt=input_text1, LLM_type=LLM_type, json_output=True, model_name=model_name)
    else:
        flag = 0
        while flag == 0:


            try:
                t = call_LLM(prompt=input_text1, LLM_type=LLM_type, json_output=False, model_name=model_name)
                t = extract_json(t)
                anst = json.loads(t)
                flag = 1
            except:
                print("产生错误，重新生成（可能是json解析失败，或者是触发TPM限制）")
                anst = {}
        ans1 = anst
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": current_time,
        "filename": textname,
        "index": 1,
        "result": ans1
    }
    with open(f"Data/{textname}/{textname}_text_analysis.jsonl", "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    summary = ans1["Summary"]
    advice = ans1["Suggestions"]

    for index in range(2, total_index + 1):
        with open(f"Data/Chunks/{textname}_Large/chunk_{index:03d}.txt", "r", encoding="utf-8") as f:
            text = f.read()
        input_text = PROMPT_TEMPLATE_2.format(text=text, summary=summary, advice=advice, i=index)
        logger.info(f"Processing chunk {index}...")
        if json_mode:
            ans = call_LLM(prompt=input_text, LLM_type=LLM_type, json_output=True, model_name=model_name)
        else:
            flag = 0
            while flag == 0:
                try:
                    t = call_LLM(prompt=input_text, LLM_type=LLM_type, json_output=False, model_name=model_name)
                    t = extract_json(t)
                    ans = json.loads(t)
                    flag = 1
                except :
                    print("json格式错误，重新生成")
                    ans = {}


        summary = ans["Summary"]
        advice = ans["Suggestions"]
        log_entry = {
            "timestamp": current_time,
            "filename": textname,
            "index": index,
            "result": ans
        }
        with open(f"Data/{textname}/{textname}_text_analysis.jsonl", "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# TODO 还需要改进 不定项输入 以决定具体调用哪个LLM
def process_chunks(exd_list, textname, LLM_type, model_name, json_mode, analysis_list, index_L, index_S):
    # 寻找exd_list中是否有index_L 和 index_S 相同的元素，如果有则跳过
    if any(item["index_L"] == index_L and item["index_S"] == index_S for item in exd_list):
        logger.info(f"Data/{textname}/{textname}_extract_result.jsonl 的第{index_L}部分context 第{index_S}段抽取跳过")
        return

    logger.info(f"Data/{textname}/{textname}_extract_result.jsonl 的第{index_L}部分context 第{index_S}段抽取开始")
    with open(f"Data/Chunks/{textname}_Small/{textname}_{index_L:03d}_{index_S:03d}.txt", "r", encoding="utf-8") as f:
        text = f.read()
    summary = analysis_list[index_L - 1]["result"]["Summary"]
    advice = analysis_list[index_L - 1]["result"]["Suggestions"]
    input_text = PROMPT_TEMPLATE_3.format(input_text=text, summary=summary, advice=advice)
    if json_mode:
        ans = call_LLM(prompt=input_text, LLM_type=LLM_type, model_name=model_name, json_output=True)
    else:
        ans = call_LLM(prompt=input_text, LLM_type=LLM_type, model_name=model_name)
        ans = extract_json(ans)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": current_time,
        "filename": textname,
        "index_L": index_L,
        "index_S": index_S,
        "result": ans
    }
    append_dict_to_jsonl(f"Data/{textname}/{textname}_extract_result.jsonl", log_entry)

    # 写log
    logger.info(f"Data/{textname}/{textname}_extract_result.jsonl 的第{index_L}部分context 第{index_S}段抽取完成")


def analyze_text_small(textname, LLM_type, model_name, json_mode, analysis_list, index, works):
    small_chunks_num = count_files(f'Data/Chunks/{textname}_Small', f'{textname}_{index:03d}')

    # 已经重复抽取的就不抽取了，应对并发限制的无奈之举
    exd_list = read_jsonl_to_list(f"Data/{textname}/{textname}_extract_result.jsonl")
    with ThreadPoolExecutor(max_workers=works) as executor:
        futures = []
        for i in range(1, small_chunks_num + 1):
            future = executor.submit(process_chunks, exd_list, textname, LLM_type, json_mode=json_mode,
                                     model_name=model_name, analysis_list=analysis_list, index_L=index, index_S=i)
            futures.append(future)
        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()  # 获取任务的返回结果
            except Exception as e:
                print(f"任务执行出错: {e}")


def analyze(filename, analysis_list, index, LLM_type, model_name, json_mode, workers):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i in range(1, index + 1):
            future = executor.submit(analyze_text_small, filename, LLM_type=LLM_type,
                                     model_name=model_name, json_mode=json_mode,
                                     analysis_list=analysis_list, index=i, works=2)
            futures.append(future)


def main0_0(textname, LLM_type, model_name, json_mode, workers):
    # TODO 修
    # 检查文件是否存在
    if os.path.exists(f"Data/{textname}/{textname}_text_analysis.jsonl"):
        print(f"文件 'Data/{textname}/{textname}_text_analysis.jsonl' 存在。")
    else:
        print(f"文件 'Data/{textname}/{textname}_text_analysis.jsonl' 不存在。开始构建。。。进行large分析")
        logger.info(f"进行初始分析 analyze_text_large")
        analyze_text_large(textname, LLM_type=LLM_type, model_name=model_name, json_mode=False)



    analysis_list = read_jsonl_to_list(f"Data/{textname}/{textname}_text_analysis.jsonl")

    index = len(analysis_list)

    count_all = count_txt_files(f"Data/Chunks/{textname}_Small")
    count_exd = count_lines(f"Data/{textname}/{textname}_extract_result.jsonl")

    while count_exd < count_all:
        logger.info(f"文段分析进度： {count_exd}/{count_all}")
        analyze(textname, analysis_list, index, LLM_type, model_name, json_mode=json_mode, workers=workers)
        analysis_list = read_jsonl_to_list(f"Data/{textname}/{textname}_text_analysis.jsonl")
        index = len(analysis_list)
        count_exd = count_lines(f"Data/{textname}/{textname}_extract_result.jsonl")

    logger.info(f"抽取完毕")

if __name__ == '__main__':
    textname = "gmzz_1"
    LLM_type = "Silicon"
    model_name = "Pro/deepseek-ai/DeepSeek-V3"
    worker = 3
    json_mode = False
    reset_jsonl_file(f"Data/{textname}/{textname}_extract_result.jsonl")

    main0_0(textname, LLM_type, model_name, json_mode, worker)






