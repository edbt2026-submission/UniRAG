import json
import re

from openai import OpenAI

from llm_core import call_LLM
from logger_config import setup_logger
from tools import read_jsonl_to_list, count_txt_files, reset_jsonl_file, sort_jsonl_file, append_dict_to_jsonl, \
    extract_json, count_lines, find_elements_in_A_not_in_B, remove_elements_inplace, write_jsonl_file

from step2_0_go_merge import step2_0_main

from concurrent.futures import ThreadPoolExecutor, as_completed
# 日志文件路径
log_file_path = 'UniRAG.log'
# 设置日志记录器
logger = setup_logger(log_file_path)

PROMPT_TEMPLATE = """Given a piece of text and two entities extracted from the text, please help me determine whether these two entities refer to the same person or thing.

The text is as follows:
{text}

The first entity:
{entity1}

The second entity:
{entity2}

You need to answer YES or NO. YES means you believe they refer to the same person or thing, and NO means you believe they do not.
Your answer should be in JSON format, with an example output as follows:
{{
    "result": <Your answer, YES or NO>
}}"""


def merge_check(**kwargs):
    filename = kwargs["filename"]
    ex_list = kwargs["ex_list"]
    merge_list = kwargs["merge_list"]
    index = kwargs["index"]
    outputfile = kwargs.get("outputfile", f"Data/{filename}/{filename}_final_merge.jsonl")
    final_list = kwargs.get("final_list", [])

    logger.info(f"开始检查{filename} 的第{index-1}和第{index}部分 的合并结果")
    # 检查是否已经验证过
    if any(item["index"] == index for item in final_list):
        logger.info(f"跳过{index}")
        return

    LLM_Type = kwargs.get("LLM_Type", "DeepSeek")
    model_name = kwargs.get("model_name", "deepseek-chat")
    json_output = kwargs.get("json_output", False)
    entities_A = ex_list[index - 1]
    entities_B = ex_list[index]
    with open(f'Data/Chunks/{filename}_Small/{filename}_{entities_A["index_L"]:03d}_{entities_A["index_S"]:03d}.txt', 'r', encoding='utf-8') as file:
        chunk_A = file.read()
    with open(f'Data/Chunks/{filename}_Small/{filename}_{entities_B["index_L"]:03d}_{entities_B["index_S"]:03d}.txt', 'r', encoding='utf-8') as file:
        chunk_B = file.read()

    text = chunk_A + chunk_B
    merge_entities = merge_list[index - 1]["Merge"]

    if len(merge_entities) == 0:
        logger.info(f"{index}和{index + 1}之间不需要合并实体")
        final_merge = {"index": index, "Merge": []}
        append_dict_to_jsonl(outputfile, final_merge)

    else:
        ab_pairs = []
        for item in merge_entities:
            for key, value in item.items():
                if key.startswith('A') and value.startswith('B'):
                    a_num = int(re.search(r'\d+', key).group())
                    b_num = int(re.search(r'\d+', value).group())
                    if a_num <= len(entities_A['EX']) and 'entity' in entities_A['EX'][a_num - 1] and b_num <= len(entities_B['EX']) and 'entity' in entities_B['EX'][b_num - 1]:
                        ab_pairs.append([a_num, b_num])
        exA = ex_list[index - 1]["EX"]
        exB = ex_list[index]["EX"]

        same_entities = []
        for pair in ab_pairs:
            a = exA[pair[0] - 1]
            b = exB[pair[1] - 1]
            same_entities.append([a, b])

        final_merge = {"index": index, "Merge": []}
        for item in same_entities:

            # 检查item[0]["entity"] 和 item[1]["entity"] 是否存在
            if 'entity' not in item[0] or 'entity' not in item[1]:
                print("不存在！！")
                print(index)
                print(same_entities)
                continue

            data = {
                "entity1": item[0]["entity"],
                "entity2": item[1]["entity"],
                "text": text
            }

            filled_text = PROMPT_TEMPLATE.format(**data)

            if json_mode:
                ans = call_LLM(prompt=filled_text, LLM_type=LLM_Type, model_name=model_name, json_output=json_output)
            else:
                flag = 0
                while flag == 0:
                    json_str = call_LLM(prompt=filled_text, LLM_type=LLM_Type, model_name=model_name, json_output=json_output)
                    json_str = extract_json(json_str)
                    try:
                        ans = json.loads(json_str)
                        flag = 1
                    except:
                        print("json格式错误或者达到TPM限制, 正在重新生成回答")

            if ans["result"] == "YES":
                final_merge["Merge"].append(item)
        with open(outputfile, 'a', encoding='utf-8') as f:
            f.write(json.dumps(final_merge, ensure_ascii=False) + '\n')
        logger.info(f"{filename} 的第{index-1}和第{index}部分 的合并结果已经检查完成")


def count_merge_nodes(filename):
    merge_list = read_jsonl_to_list(filename)
    count = 0
    for item in merge_list:
        count += len(item["Merge"])
    return count


def step2_1_main(textname, LLM_Type, model_name, json_mode, worker):
    logger.info("step2_1_验证merge合法性.py 开始执行")

    filename = f"Data/{textname}/{textname}_merge.jsonl"
    original_count = count_merge_nodes(filename)
    print(f"{filename} 原始合并节点数：{original_count}")
    ex_name = f"Data/{textname}/{textname}_EX.jsonl"
    entities_list = read_jsonl_to_list(filename)
    ex_list = read_jsonl_to_list(ex_name)
    sorted_list = sorted(entities_list, key=lambda x: x.get('index', 0))
    total_index = len(entities_list)
    outputfile = f"Data/{textname}/{textname}_final_merge.jsonl"
    final_list = read_jsonl_to_list(outputfile)

    # 创建一个线程池，最大线程数为10
    with ThreadPoolExecutor(max_workers=worker) as executor:
        futures = []
        for i in range(1, total_index + 1):
            # 提交任务到线程池
            # print(f"开始处理文件：{filename} 的第{i}和第{i + 1}部分")
            kwargs = {
                "filename": textname,
                "ex_list": ex_list,
                "merge_list": sorted_list,
                "final_list": final_list,
                "index": i,
                "outputfile": outputfile,
                "LLM_Type": LLM_Type,
                "model_name": model_name,
                "json_output": json_mode
            }
            future = executor.submit(merge_check, **kwargs)
            futures.append(future)

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()  # 获取任务的返回结果
            except Exception as e:
                print(f"任务执行出错: {e}")

    sort_jsonl_file(outputfile, outputfile)
    final_count = count_merge_nodes(outputfile)
    print(f"{filename} 最终合并节点数：{final_count}")


if __name__ == '__main__':
    textname = 'gmzz_1'
    # LLM_Type = 'Silicon'
    # model_name = 'Pro/deepseek-ai/DeepSeek-V3'
    # json_mode = False
    # worker = 4

    LLM_Type = 'GLM'
    model_name = 'glm-4-air'
    json_mode = True
    worker = 10

    filename = f"Data/{textname}/{textname}_merge.jsonl"
    ex_name = f"Data/{textname}/{textname}_EX.jsonl"

    outputfile = f"Data/{textname}/{textname}_final_merge.jsonl"
    reset_jsonl_file(outputfile)

    count_all = count_lines(filename)
    count_merged = count_lines(outputfile)
    while count_all > count_merged:
        step2_1_main(textname, LLM_Type, model_name, json_mode, worker)
        count_merged = count_lines(outputfile)
        merge_list = read_jsonl_to_list(filename)
        final_merge_list = read_jsonl_to_list(outputfile)
        diff_list = find_elements_in_A_not_in_B(merge_list, final_merge_list)
        print("产生的错误个数："+str(len(diff_list)))

        for each in diff_list:
            index = each["index"]
            print("删除第"+str(index)+"个")
            remove_elements_inplace(merge_list, index)

        write_jsonl_file(filename, merge_list)
        step2_0_main(textname, LLM_Type, model_name, json_mode, worker)

    logger.info("step2_1_验证merge合法性.py 执行完毕")