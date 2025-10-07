import json

from tools import read_jsonl_to_list


def json_to_markdown(json_data):
    markdown = ""

    # 提取查询
    query = json_data.get("query", "")
    markdown += f"# {query}\n\n"

    # 遍历每个模型
    for model, responses in json_data.items():
        if model == "query":
            continue  # 跳过查询部分

        markdown += f"## {model}\n\n"

        # 遍历每种回答类型
        for response_type, content in responses.items():
            markdown += f"### {response_type}\n\n"
            markdown += f"```\n{content}\n```\n\n"

    return markdown

if __name__ == '__main__':
    list = read_jsonl_to_list("QA_result.jsonl")

    for i in list:
        markdown = json_to_markdown(i)
        with open(f"QA_result.md", "a", encoding="utf-8") as f:
            f.write(markdown)
            f.write("\n")