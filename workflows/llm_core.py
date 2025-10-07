import datetime
import json
import threading
import time
import requests

from openai import OpenAI
from tools import extract_json, append_dict_to_jsonl
# from logger_config import setup_logger
# # 日志文件路径
# log_file_path = 'MyRAG.log'
# # 设置日志记录器
# logger = setup_logger(log_file_path)
save_slot = "llm_call6"
save_slot_embedding = "embedding_call001"

SILICONCLOUD_API_KEYS = ["sk-urmkueripvadmnsfoeswacjufqukqdogopjpctxxhqijxofb", "sk-vzagbwfrmogslyxfbznxepubqmpihefggzxovxjivirjyxbw", "sk-kyyxisvhwzjkichrbcknswrrlkjnvcjjdxhlvsrvtkqedrye"]

DEEPSEEK_API_KEY = "sk-8635b3d207b846918a759b4bbd7d1151"
QWEN_API_KEY = "sk-sQtm2e7Ce4"
GLM_API = "8932540170944a129b087d20d5ef7bb0.R8XBsih8OKlLA4HV"
HW_API = "vkwGxIl2OzP_YSMMNKdq_dAxGty8_JY2YyzaCv2N9IT-TXRlFJY2pVHsDnPK7rO_cCIY1LxojPgNnR5xjpu0nA"

class RoundRobin:
    def __init__(self, items):
        self.items = items
        self.index = 0
        self.lock = threading.Lock()

    def next(self):
        with self.lock:
            item = self.items[self.index]
            self.index = (self.index + 1) % len(self.items)
            return item

# 初始化轮询器
silicon_round_robin = RoundRobin(SILICONCLOUD_API_KEYS)

def get_client(llm_name: str):
    if llm_name == "Silicon":
        client = OpenAI(api_key=silicon_round_robin.next(), base_url="https://api.siliconflow.cn/v1")
    elif llm_name == "DeepSeek":
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
    elif llm_name == "Qwen":
        client = OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    elif llm_name == "GLM":
        client = OpenAI(api_key=GLM_API, base_url="https://open.bigmodel.cn/api/paas/v4")
    elif llm_name == "X":
        client = OpenAI(api_key="AAA", base_url="http://10.196.83.122:10007/v1")
    elif llm_name == "HW":
        client = OpenAI(api_key=HW_API, base_url="https://infer-modelarts-cn-southwest-2.myhuaweicloud.com/v1/infers/f354eacc-a2c5-43b4-a785-e5aadca988b3/v1")
    else:
        raise ValueError("Invalid LLM name")
    return client


def call_LLM(**kwargs):
    # 记录开始时间
    start_time = time.time()
    required_keys = ["prompt", "LLM_type"]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Missing required argument: {key}")
    LLM_type = kwargs.get("LLM_type")
    if LLM_type == "Silicon":
        model_name = kwargs.get("model_name", "Pro/deepseek-ai/DeepSeek-V3")
        client = get_client(LLM_type)
    elif LLM_type == "DeepSeek":
        model_name = kwargs.get("model_name", "deepseek-chat")
        client = get_client(LLM_type)
    elif LLM_type == "Qwen":
        model_name = kwargs.get("model_name", "qwen-max-0125")
        client = get_client(LLM_type)
    elif LLM_type == "GLM":
        model_name = kwargs.get("model_name", "glm-4-air")
        client = get_client(LLM_type)
    elif LLM_type == "X":
        model_name = kwargs.get("model_name", "R1-Qwen-7B")
        client = get_client(LLM_type)
    elif LLM_type == "HW":
        model_name = kwargs.get("model_name", "DeepSeek-V3")
        client = get_client(LLM_type)
    else:
        raise ValueError("Invalid LLM name")

    # 检查入参中是否需要json输出
    if kwargs.get("json_output", False):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'user',
                 'content': kwargs.get("prompt")}
            ],
            stream=False,
            response_format={
                'type': 'json_object'
            }
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'user',
                 'content': kwargs.get("prompt")}
            ],
            stream=False
        )
    # 记录结束时间
    end_time = time.time()

    # 计算耗时 保留两位小数
    elapsed_time = round(end_time - start_time, 2)
    prompt = kwargs.get("prompt")
    # 写入本地jsonl文件记录调用
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": current_time,
        "cost_time": elapsed_time,
        "token": getattr(response.usage, 'total_tokens', 0),
        "LLM_type": LLM_type,
        "model_name": model_name,
        "speed": round(response.usage.completion_tokens / elapsed_time, 2) if hasattr(response.usage, 'completion_tokens') else 0,
        "prompt": kwargs.get("prompt"),
        "response": response.choices[0].message.content
    }
    with open(f"{save_slot}.jsonl", "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    result_text = response.choices[0].message.content


    # 检查入参中是否需要json输出
    if kwargs.get("json_output", False):
        try:
            result_json = json.loads(result_text)
            # 写入logger
            # logger.info(f"{current_time} {LLM_type} {model_name} {prompt} {result_json}")
            return result_json
        except json.JSONDecodeError:
            try_result = extract_json(result_text)
            try:
                result_json = json.loads(try_result)
                # 写入logger
                # logger.info(f"经过修复，JSON 解析成功 {current_time} {LLM_type} {model_name} {prompt} {result_json}")
                return result_json
            except json.JSONDecodeError:
                # logger.error(f"JSON 最终还是解析失败:{current_time} {LLM_type} {model_name} {prompt} {result_text}")
                return "JSON decode error " + str(current_time)
    else:
        return result_text

def get_embedding(**kwargs):
    start_time = time.time()
    required_keys = ["text", "Embedding_type"]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Missing required argument: {key}")
    Embedding_type = kwargs.get("Embedding_type")
    if Embedding_type == "Silicon":
        model_name = kwargs.get("model_name", "Pro/BAAI/bge-m3")
        client = get_client(Embedding_type)
    else:
        raise ValueError("Invalid Embedding name")
    response = client.embeddings.create(
        model=model_name,
        input=kwargs.get("text")
    )
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": current_time,
        "cost_time": elapsed_time,
        "token": getattr(response.usage, 'total_tokens', 0),
        "Embedding_type": Embedding_type,
        "model_name": model_name,
        "speed": round(response.usage.total_tokens / elapsed_time, 2) if hasattr(response.usage, 'completion_tokens') else 0,
        "text": kwargs.get("text"),
        "response": response.data[0].embedding
    }
    append_dict_to_jsonl(f"{save_slot_embedding}.jsonl", log_entry)

    return response.data[0].embedding

def rerank(query, documents, model="BAAI/bge-reranker-v2-m3", top_n=10, return_documents=False, max_chunks_per_doc=1024,
           overlap_tokens=80, token=silicon_round_robin.next()):
    """
    调用 SiliconFlow 的 Reranker 模型进行文档重排序。

    :param query: 查询字符串
    :param documents: 文档列表，例如 ["苹果", "香蕉", "水果", "蔬菜"]
    :param model: 使用的模型名称，默认为 "BAAI/bge-reranker-v2-m3"
    :param top_n: 返回的文档数量，默认为 10
    :param return_documents: 是否返回文档内容，默认为 False
    :param max_chunks_per_doc: 每个文档的最大分块数，默认为 1024
    :param overlap_tokens: 分块之间的重叠 token 数，默认为 80
    :param token: API 的认证 token
    :return: API 的响应结果
    """
    url = "https://api.siliconflow.cn/v1/rerank"

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": return_documents,
        "max_chunks_per_doc": max_chunks_per_doc,
        "overlap_tokens": overlap_tokens
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def read_text_from_file(path):
    """从指定路径的文件读取文本内容"""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"错误：文件 '{path}' 未找到。")
        return None
    except Exception as e:
        print(f"发生错误：{e}")
        return None


if __name__ == '__main__':
    # 测试代码
    # prompt = "首先请你先阅读一段文本：\n说起人类这玩意儿，总是夸大吹捧自己的功勋，瞧他们的嘴脸，仿佛历史全是由他们一手创造，委实滑稽之至，贻笑大方。就算借助狸猫的力量，这些一吹就跑的小小人类又能有何作为？一切天灾和动乱皆由我等魔道中人控制，国家的命运尽握吾等手中。抬头仰望这城市四周的山巅吧!好好对居住天界的我们心存敬畏吧!\n\n\t    有人狂傲地撂下这等豪语。不用说也知道，是天狗。人类在街上生活，狸猫在地上爬行，天狗在天空飞翔。迁都平安城后，人类、狸猫、天狗，三足鼎立。他们转动这城市的巨大车轮。天狗对狸猫说教，狸猫迷惑人类，人类敬畏天狗。天狗又掳走人类，人类把狸猫煮成火锅，狸猫设圈套引诱天狗。就这样，车轮不断转动。望着那转动的车轮，乐趣无穷。而我就是众人口中的狸猫。然而我不屑于当只平庸的狸猫，我仰慕天狗，也喜欢模仿人类。因此，我的日常生活精采得教人眼花缭乱，一点都不无聊。第一章 纳凉露台女神\n\n\t  有位退休的天狗住在出町商店街北边一栋名叫“树形住宅”的公寓里。他鲜少外出。总是随手将商店街买来的食材丢进锅里，煮成一锅可怕的热粥，以此果腹延命。他老得吓人，排斥洗澡的程度古今无人能出其右，所幸他那干瘪得犹如鱿鱼干的皮肤不管再怎么使劲搓揉也搓不出污垢。尽管一个人什么事也办不了，他那高傲的自尊却好比秋日晴空那般高不可攀。他昔日自诩足以任意操弄国家命运的神通力，早已丧失多年。他“性”致勃勃，但享受爱情生活的能力也已丧失良久。他总是一脸心有不甘独酌红玉波特酒（注：明治时代贩售的甜味红酒。为鸟井商店的产品。后来改名为“红玉甜酒”。）。只见他浅尝醇酒，道起昔日愚蠢的人类你来我往的战乱，本以为他谈的是幕末纷争，孰料竟是应仁之乱；以为说的是应仁之乱，没想到竟是平家的衰败；以为他讲的是平家的衰败，结果却谈到幕末的种种。简言之，根本就杂乱无章。他不像拥有血肉之躯的生物，反倒与化石有几分相像。每个人都诅咒他早点变成石头。我们都喊他红玉老师。这位天狗，正是我的恩师。○\n\n\t    住在京都的狸猫都是向天狗学习读写算术、变身术、辩论术、向貌美少女搭讪的技巧等等。京都住有许多天狗，门派林立，当中以鞍马山的鞍马天狗名气最响，据说个个都是菁英。不过我们如意岳的红玉老师也不遑多让，同样远近驰名；老师有个威风凛凛的名号，人称“如意岳药师坊”。如今一切已成过往，但想当年红玉老师还曾借用大学教室开班授课呢。位于校舍角落的昏暗阶梯教室里站满徒子徒孙，老师在讲台前尽情施展天狗本领，威风不可一世。当时老师浑身散发着货真价实的威严，学生根本不敢有任何意见。至于他是因为趾高气昂以致威严十足，还是因为威严十足才显得趾高气昂?这种没意义的怀疑，老师以不容分说的气势压下了。由此可证，他身上的是货真价实的威严。从前，老师总是身穿没有一丝皱褶的笔挺西装，板着脸，说话时总是眼望窗外的树丛。我回想起他令人怀念的身影。我瞧不起你们——这句话老师说了不下百遍。他还说，我瞧不起的不只你们，我瞧不起自己以外的任何人。在空中飞翔，恣意刮起旋风，看上的姑娘掳了就走，唾沫吐尽世上万物。那是红玉老师不可一世的过去。有谁能料到老师如今竟落魄潦倒，只能屈身于商店街的小公寓。多年以来，我们狸猫一族接受红玉老师教导，我也不例外，入门拜他为师。回想过去，我总是挨老师骂。思忖挨骂的原因，大概是我没能认真修行，为狸猫一族贡献一己之力。我太骄纵任性，只想走自己的路，一心憧憬崇高地位。然而老师坐拥崇高宝座，却不乐见其他人也登上高位。尽管如此，当时我很希望能和老师一样。事到如今，一切已成往事。○\n\n\t    拜访红玉老师那天，我先绕去了山町的商店街，街上满是购物人潮好不热闹，人类臭味熏天。我买好红玉波特酒、卫生纸、棉花棒和便当，走进一路向北延伸的小巷。那是衹园祭已经结束、七月底的某个黄昏。我变身成一名可爱的女高中生。我从小就只有变身术拿手，由于老是变个不停，挨骂成了家常便饭。近年来，随着狸猫的变身能力普遍低落，逐渐兴起一股奇怪的风潮，主张就算是狸猫也不能随意变身。简直是无聊透顶。恣意施展得天独厚的天赋愉快度日，有什么不对?我之所以变身成青春可爱的少女，还不是为了老师。\n我现在让两个助手对这段文本的上下两部分分别进行了实体抽取，即从文本中识别这些类型的所有实体对象。\n第一组，负责上半段文本抽取的助手A抽取结果如下：\n[{'name': '人类', 'type': '种族', 'description': '生活在街道上的生物，与狸猫和天狗共同居住在城市中，彼此之间存在复杂的互动关系。'}, {'name': '狸猫', 'type': '种族', 'description': '在地上爬行的生物，擅长变身，与天狗和人类之间有着错综复杂的关系。'}, {'name': '天狗', 'type': '种族', 'description': '在天空飞翔的生物，曾经拥有操控国家命运的能力，如今生活在城市中，与人类和狸猫形成三足鼎立的局面。'}, {'name': '红玉老师', 'type': '角色', 'description': '退休的天狗，住在出町商店街的‘树形住宅’公寓中，过去曾拥有强大的神通力，如今过着落魄的生活，依然保持高傲的自尊。'}, {'name': '出町商店街', 'type': '地点', 'description': '红玉老师居住的地方，位于京都的一个商店街。'}, {'name': '树形住宅', 'type': '地点', 'description': '位于出町商店街北边的一栋公寓，红玉老师的居住地。'}, {'name': '红玉波特酒', 'type': '物品', 'description': '红玉老师常饮的一种甜味红酒，原为明治时代的鸟井商店产品，后改名为‘红玉甜酒’。'}]\n第二组，负责上半段文本抽取的助手B抽取结果如下：\n[{'name': '京都', 'type': '地点', 'description': '日本的古都，故事发生的奇幻城市，狸猫、天狗和人类共同生活在此。'}, {'name': '下鸭矢三郎', 'type': '角色', 'description': '故事的主角，一只年轻的狸猫，擅长变身术，是红玉老师的学生。'}, {'name': '红玉老师', 'type': '角色', 'description': '一位退休的天狗，曾以‘如意岳药师坊’闻名，现已落魄，仍然迷恋弁天，是矢三郎的师父。'}, {'name': '弁天', 'type': '角色', 'description': '由人类转变成天狗的女性，曾被红玉老师绑架并培养，但最终背叛了他。'}, {'name': '鞍马天狗', 'type': '角色', 'description': '鞍马山的知名天狗，被认为是一群精英。'}, {'name': '如意岳药师坊', 'type': '角色', 'description': '红玉老师过去的名号，象征着他在天狗界的地位和威严。'}, {'name': '红玉波特酒', 'type': '物品', 'description': '矢三郎为红玉老师购买的物品之一，可能是老师喜爱的饮品。'}, {'name': '衹园祭', 'type': '事件', 'description': '京都的著名节日，原文中提到祭典已经结束。'}, {'name': '狸猫', 'type': '物种', 'description': '故事中的一类生物，擅长变身术，与天狗和人类共同生活在京都。'}, {'name': '天狗', 'type': '物种', 'description': '故事中的一类生物，拥有飞行和操控旋风的能力，与狸猫和人类共同生活在京都。'}]\n我现在需要你做的事情是找出两组抽取结果中，在文中指向的是同一个人物或事物的实体对象\n你需要重点关注两组中的人称代词是否在另一组中有相同对象的不同名称，比如“我”，当然文中也可能会出现人称代词没有对应的名称的情况，这种时候不需要合并这些人称代词\n你只需要合并并输出你认为相同的实体，对于不同的实体，你不需要输出\n你的输出格式为jsonl，具体格式为：\n[\n    {\n        \"AssistantA\": <助手A所抽取的实体名称>,\n        \"AssistantB\": <助手B所抽取的实体名称>\n    },\n    {\n        \"AssistantA\": <助手A所抽取的实体名称>,\n        \"AssistantB\": <助手B所抽取的实体名称>\n    },\n    ......\n]\n\n你的jsonl:\n\n\n"
    text = read_text_from_file("./Data/test/test.txt")
    prompt = f"""
    现在有一段文本如下：
    {text}
    
    我现在想对这一段文本进行知识图谱的构建，纵观全局，我想先抽取出一些核心实体类型，以后可以依据这些类型进行抽取。
    类型名称只需要中文
    你有什么好的建议吗？
    
    你的输出以json格式输出，例子如下：
    [
        {{
            "type_name": "<类型1>",
            "example": ["<例1>", "<例2>"],
            "explanation": "<解释>"
        }},
        {{
            "type_name": "<类型2>",
            "example": ["<例1>", "<例2>"],
            "explanation": "<解释>"
        }}
    ]
    """.strip()
    # result = call_LLM(prompt=prompt, LLM_type="Silicon", model_name="Pro/deepseek-ai/DeepSeek-V3") # model_name="Pro/deepseek-ai/DeepSeek-V3"
    result = get_embedding(text=prompt, Embedding_type="Silicon")
    print(result)

