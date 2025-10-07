import json
import re
import threading

from openai import OpenAI

from llm_core import call_LLM
from logger_config import setup_logger
from tools import read_jsonl_to_list, count_txt_files, reset_jsonl_file, extract_json, count_lines, sort_jsonl_file

from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "sk-vzagbwfrmogslyxfbznxepubqmpihefggzxovxjivirjyxbw"
DEEPSEEK_API_KEY = "sk-8635b3d207b846918a759b4bbd7d1151"
# client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
# 日志文件路径
log_file_path = 'UniRAG.log'
# 设置日志记录器
logger = setup_logger(log_file_path)


PROMPT_TEMPLATE_S = '''I have now asked two assistants to perform entity extraction on the upper and lower parts of a text, respectively, identifying all entity objects of these types from the text.

What I need you to do now is to find entity objects in the two sets of extraction results that refer to the same person or thing in the text.
You need to focus on whether pronouns in one group have corresponding names in the other group, such as "I." Of course, there may also be cases where pronouns in the text do not have corresponding names. In such cases, you do not need to merge these pronouns.
You only need to merge and output entities that you believe are the same. For different entities, you do not need to output them.
Your output should be in JSON format, like this:
[
    {{<The entity number of the same entity in the first group, such as A1, A2, etc.>:<The entity number of the same entity in the second group, such as B1, B2, etc.>, "reason":<The reason why you think they are the same entity object>}},
    {{<The entity number of the same entity in the first group, such as A1, A2, etc.>:<The entity number of the same entity in the second group, such as B1, B2, etc.>, "reason":<The reason why you think they are the same entity object>}},
    ······
]

Your output language needs to be consistent with the language of the content.

Note: nameA must be an entity that appears in Assistant A's extraction results, and nameB must also be an entity that appears in Assistant B's extraction results.

If you think there are no identical entities in the two groups, simply output an empty list [].

Your current task is as follows:
The complete original text is as follows:
{text}

The first group, Assistant A, responsible for extracting the upper part of the text, has the following extraction results:
{entities1}

The second group, Assistant B, responsible for extracting the upper part of the text, has the following extraction results:
{entities2}

Your extraction result:
'''


def entities_merge(filename: str, index: int, outputfile: str, entities_list: list, LLM_Type: str, model_name: str, json_mode: bool, merged_list: list):
    # print(index)
    logger.info(f"开始处理文件：{filename} 的第{index}和第{index+1}部分")

    # 检查是否已经合并过了
    if any(item["index"] == index for item in merged_list):
        logger.info(f"跳过{index}")
        return


    entities_A = entities_list[index - 1]
    entities_B = entities_list[index]

    groupA = []
    groupB = []

    for item in entities_A["EX"]:
        if "entity" in item:
            inner_index = item["inner_index"]
            new_key = f"A{inner_index}"
            new_item = {new_key: item["entity"]}
            groupA.append(new_item)

    for item in entities_B["EX"]:
        if "entity" in item:
            inner_index = item["inner_index"]
            new_key = f"B{inner_index}"
            new_item = {new_key: item["entity"]}
            groupB.append(new_item)

    with open(f'Data/Chunks/{filename}_Small/{filename}_{entities_A["index_L"]:03d}_{entities_A["index_S"]:03d}.txt', 'r', encoding='utf-8') as file:
        chunk_A = file.read()

    with open(f'Data/Chunks/{filename}_Small/{filename}_{entities_B["index_L"]:03d}_{entities_B["index_S"]:03d}.txt', 'r', encoding='utf-8') as file:
        chunk_B = file.read()

    # 定义要填充的数据
    data = {
        "entities1": groupA,
        "entities2": groupB,
        "text": chunk_A + chunk_B
    }

    # 使用字典填充模板
    filled_text = PROMPT_TEMPLATE_S.format(**data)

    if json_mode:
        # ans = call_LLM(prompt=filled_text, LLM_type="GLM", model_name="glm-4-air", json_output=True)
        ans = call_LLM(prompt=filled_text, LLM_type=LLM_Type, model_name=model_name, json_output=True)
    else:
        ans_str = call_LLM(prompt=filled_text, LLM_type=LLM_Type, model_name=model_name, json_output=False)
        ans_str = extract_json(ans_str)
        ans = json.loads(ans_str)    # # 将结果追加到文件中
    output = {
        "index": index,
        "Merge": ans
    }
    with open(outputfile, 'a', encoding='utf-8') as f:
        f.write(json.dumps(output, ensure_ascii=False) + '\n')
    logger.info(f"{filename} 的第{index-1}和第{index}部分 的合并结果已经生成")


def step2_0_main(textname, LLM_Type, model_name, json_mode, worker):
    logger.info("step2_0_go_merge.py 开始执行")
    filename = f"Data/{textname}/{textname}_EX.jsonl"
    outputfile = f"Data/{textname}/{textname}_merge.jsonl"
    entities_list = read_jsonl_to_list(filename)
    sorted_list = sorted(entities_list, key=lambda x: x.get('index', 0))
    total_index = len(entities_list)
    merged_list = read_jsonl_to_list(outputfile)

    # 创建一个线程池，最大线程数为10
    with ThreadPoolExecutor(max_workers=worker) as executor:
        futures = []
        for i in range(1, total_index):
            # 提交任务到线程池
            future = executor.submit(entities_merge, textname, i, outputfile, sorted_list, LLM_Type, model_name,
                                     json_mode, merged_list)
            futures.append(future)

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()  # 获取任务的返回结果
            except Exception as e:
                print(f"任务执行出错: {e}")





if __name__ == '__main__':
    textname = 'gmzz_1'
    filename = f"Data/{textname}/{textname}_EX.jsonl"
    outputfile = f"Data/{textname}/{textname}_merge.jsonl"
    json_mode = False

    # 注意此行的状态
    # reset_jsonl_file(outputfile)


    LLM_Type = "Silicon"
    model_name = "Pro/deepseek-ai/DeepSeek-V3"
    worker = 2

    count_all = count_lines(filename)
    count_merged = count_lines(outputfile)
    while count_all > count_merged + 1:
        step2_0_main(textname, LLM_Type, model_name, json_mode, worker)
        count_all = count_lines(filename)
        count_merged = count_lines(outputfile)

    # 最后排个序
    sort_jsonl_file(outputfile, outputfile)