# UniRAG 项目说明 🚀

本项目是一个基于知识图谱的检索增强生成 (RAG) 框架。它通过一系列精细的步骤，将原始文本文档处理、分析、并构建成一个 Neo4j 知识图谱。最终结合向量数据库 Faiss，实现高质量的问答。

相较于传统的朴素 RAG，本项目的优势在于利用知识图谱进行深度的关联性检索，即使在使用较小参数量的语言模型（如 Qwen2.5-7B-Instruct）时，也能获得显著优于传统 RAG 的回答质量。

---

## 📚 目录

* [环境安装](#-环境安装-installation)
* [使用流程](#-使用流程-workflow)
* [项目结构（可选）](#-项目结构-可选)

---

## 🛠️ 环境安装 (Installation)

在开始之前，请确保您的系统已安装 **Python (推荐 3.10)**、**Anaconda (或 Miniconda)** 以及 **Neo4j 数据库**。

### 1. 克隆项目

```bash
git clone https://github.com/edbt2026-submission/UniRAG.git
```

### 2. 创建并激活 Conda 环境

我们强烈建议使用 Conda 来管理项目环境，以避免包版本冲突。

```bash
# 创建一个名为 UniRAG 的新环境，并指定 Python 版本
conda create -n UniRAG python=3.10

# 激活该环境
conda activate UniRAG
```

### 3. 安装依赖

项目所需的所有依赖项都已记录在 `requirements.txt` 文件中。

```bash
# 使用 pip 安装所有依赖
pip install -r requirements.txt
```

### 4. 配置 Neo4j 数据库

在开始构建图谱之前，您需要确保 Neo4j 服务已启动，并正确配置连接信息。

> **重要提示**:
> 请打开 `Neo4j_tools.py` 文件，根据您本地 Neo4j 数据库的实际情况，修改以下连接配置：
> * URI (例如: "bolt://localhost:7687")
> * 用户名 (例如: "neo4j")
> * 密码 (例如: "your_password")

---

### 5. 配置 API_KEY

在初次构建时，可以尝试使用小文本放在对应目录下，填写LLM平台的API_KEY信息进行使用，例如deepseek，或是silicon。

## 🚀 使用流程 (Workflow)

请严格按照以下步骤顺序执行脚本，以确保数据处理流程的正确性。

### 第一步：准备文件

1.  将您需要处理的 `.txt` 格式的原始文本文档放置在项目的根目录下。

### 第二步：文本处理与节点融合

此阶段负责将原始文本拆分、分析、抽取实体，并进行初步的节点融合。

1.  **文档拆分**:
    * 运行脚本: `python step0-0-文档拆分器.py`
    * **作用**: 将整个文档切分成更小的、易于处理的文本块。

2.  **文段分析与实体抽取**:
    * 运行脚本: `python step0-1-0-文段分析.py`
    * **注意**:
        * 此步骤依赖 DeepSeek 的官方 API 进行实体和关系的抽取。请确保您已配置好相关 API Key。
        * 在代码中，请根据需要解除 `analyze_text_large` 函数的注释。

3.  **检查并清洗分析结果**:
    * 运行脚本: `python step0-1-1-额外转换.py`
    * **作用**: 此脚本会验证上一步生成的 JSON 结果是否合法。
        * 它会将抽取结果中格式错误的条目删除，并生成 `{filename}_extract_result.jsonl` 文件。
        * **请检查此脚本的输出**。如果提示有错误条目被删除，您需要**重新运行 `step0-1-0-文段分析.py`** 来补充缺失的数据。
        * 最终，请确保 `{filename}_extract_result.jsonl` 文件中的 `entities` 字段都已成功转换为字典格式。

4.  **格式转换**:
    * 运行脚本: `python step0-2-结点文件转换.py`
    * **作用**: 将分析结果转换为准备进行节点融合的中间格式，并保存在 `{filename}_EX.jsonl` 文件中。

5.  **局部节点融合**:
    * 运行脚本: `python step2_0_go_merge.py`
    * **作用**: 根据实体间的关系进行初步的融合计算。
    * **注意**: 此步骤计算量较大，运行时间可能较长，请耐心等待。结果保存在 `{filename}_merge.jsonl` 中。

6.  **融合结果合法性验证**:
    * 运行脚本: `python step2-1-验证merge合法性.py`
    * **作用**: 排除上一阶段中可能产生的不合法融合，并生成最终版的 `{filename}_final_merge.jsonl`。

7.  **创建融合索引**:
    * 运行脚本: `python step2-2-结点融合.py`
    * **作用**: 为融合后的节点创建索引，准备写入图数据库。结果保存在 `{filename}_EX_pro.jsonl` 中。

### 第三步：Neo4j 图谱构建

此阶段将处理好的数据正式写入 Neo4j 数据库，并进行图层面的融合。

1.  **节点入库**:
    * 运行脚本: `python step3-0-结点入Neo4j.py`
    * **作用**: 将所有基础节点数据写入 Neo4j。运行后可登录 Neo4j Browser 查看节点是否创建成功。
    * **工具**: 同文件夹下的 `DeleteALL.py` 可用于一键清空数据库，方便调试。

2.  **局部融合关系生成**:
    * 运行脚本: `python step3-1-局部融合.py`
    * **作用**: 在本地计算并生成需要进行融合的实体组列表 (`{filename}_group_fusion.jsonl`)。

3.  **局部融合关系入库**:
    * 运行脚本: `python step3-2-局部融合写进Neo4j.py`
    * **作用**: 将上一步生成的融合关系写入 Neo4j，为 `Group` 类型的节点添加属性。

4.  **全局融合关系计算**:
    * 运行脚本: `python step3-3-全局融合.py`
    * **作用**: 在整个图谱层面进行跨实体组的全局融合计算，生成 `_group_before_global.jsonl` 文件。

5.  **全局融合关系入库**:
    * 运行脚本: `python step3-4-全局融合写进Neo4j.py`
    * **作用**: 将全局融合关系写入 Neo4j，完成图谱的最终构建。

### 第四步：向量化与问答

此阶段为图谱中的知识创建向量索引，并启动问答程序。

1.  **文本向量化**:
    * 运行脚本: `python step-4-1-Embedding.py`
    * **作用**: 将图谱中的节点文本内容转换为向量表示。

2.  **构建 Faiss 向量索引**:
    * 运行脚本: `python step-4-2-Faiss.py`
    * **作用**: 使用 Faiss 构建高效的本地向量数据库索引，用于快速检索。

3.  **开始问答**:
    * 运行主程序: `python QA.py`
    * **配置**: 您可以在 `QA.py` 文件的 `main` 函数中修改和配置问答的相关参数和命令。

---

## 📂 项目结构

```
.
├── UniRAG/
│   ├── step0-0-文档拆分器.py
│   ├── step0-1-0-文段分析.py
│   ├── step0-1-1-额外转换.py
│   ├── ... (其他所有py脚本)
│   ├── Neo4j_tools.py
│   └── QA.py
├── your_document.txt
├── requirements.txt
└── README.md
```
> 