# UniRAG Project Description 🚀

This project is a Retrieval-Augmented Generation (RAG) framework based on a knowledge graph. Through a series of detailed steps, it processes, analyzes, and constructs a Neo4j knowledge graph from raw text documents. Ultimately, it integrates with the Faiss vector database to achieve high-quality question answering.

Compared to traditional naive RAG, the advantage of this project lies in its use of the knowledge graph for deep, associative retrieval. This allows it to achieve significantly better answer quality than traditional RAG, even when using smaller language models (such as Qwen2.5-7B-Instruct).

---

## 📚 Table of Contents

* [Installation](#-installation)
* [Workflow](#-workflow)
* [Project Structure (Optional)](#-project-structure)

---

## 🛠️ Installation

Before you begin, please ensure that your system has **Python (3.10 recommended)**, **Anaconda (or Miniconda)**, and **Neo4j Database** installed.

### 1. Clone the Project

```bash
git clone https://github.com/edbt2026-submission/UniRAG.git
```

### 2. Create and Activate Conda Environment

We strongly recommend using Conda to manage the project environment to avoid package version conflicts.

```
# Create a new environment named UniRAG and specify the Python version
conda create -n UniRAG python=3.10

# Activate the environment
conda activate UniRAG
```

### 3. Install Dependencies

All required dependencies for the project are listed in the `requirements.txt` file.

```
# Use pip to install all dependencies
pip install -r requirements.txt
```

### 4. Configure Neo4j Database

Before building the graph, you need to ensure the Neo4j service is running and the connection information is correctly configured.

**Important Note:**
 Please open the `Neo4j_tools.py` file and modify the following connection settings according to your local Neo4j database configuration:

- **URI** (e.g., `"bolt://localhost:7687"`)
- **Username** (e.g., `"neo4j"`)
- **Password** (e.g., `"your_password"`)

### 5. Configure API Key

For the initial build, you can try using a small text file placed in the corresponding directory.
 Fill in the `API_KEY` information for an LLM platform, such as **DeepSeek** or **Silicon**.

------

## 🚀 Workflow

Please execute the scripts in the following order to ensure the data processing pipeline runs correctly.

### Step 1: Prepare Files

Place your raw text documents in `.txt` format in the project's root directory.

------

### Step 2: Text Processing and Node Fusion

This stage is responsible for splitting the raw text, analyzing it, extracting entities, and performing initial node fusion.

#### Document Splitting

```
python step0-0-文档拆分器.py
```

**Purpose:** Splits the entire document into smaller, more manageable text chunks.

#### Text Segment Analysis and Entity Extraction

```
python step0-1-0-文段分析.py
```

**Note:**

- This step relies on DeepSeek's official API for entity and relationship extraction.
- Please ensure you have configured the relevant API Key.
- In the code, uncomment the `analyze_text_large` function as needed.

#### Check and Clean Analysis Results

```
python step0-1-1-额外转换.py
```

**Purpose:**
 This script validates the legality of the JSON results generated in the previous step.
 It will remove entries with incorrect formats from the extraction results and generate a `{filename}_extract_result.jsonl` file.

Please check the output of this script. If it reports that erroneous entries have been deleted, you need to rerun `step0-1-0-文段分析.py` to supplement the missing data.

Ultimately, ensure that the `entities` fields in the `{filename}_extract_result.jsonl` file have all been successfully converted to dictionary format.

#### Format Conversion

```
python step0-2-结点文件转换.py
```

**Purpose:** Converts the analysis results into an intermediate format ready for node fusion and saves it in the `{filename}_EX.jsonl` file.

#### Local Node Fusion

```
python step2_0_go_merge.py
```

**Purpose:** Performs initial fusion calculations based on the relationships between entities.

**Note:** This step is computationally intensive and may take a long time to run. Please be patient.
 The results are saved in `{filename}_merge.jsonl`.

#### Fusion Result Legality Validation

```
python step2-1-验证merge合法性.py
```

**Purpose:** Excludes potentially illegal fusions from the previous stage and generates the final `{filename}_final_merge.jsonl`.

#### Create Fusion Index

```
python step2-2-结点融合.py
```

**Purpose:** Creates an index for the fused nodes, preparing them to be written to the graph database.
 The results are saved in `{filename}_EX_pro.jsonl`.

------

### Step 3: Neo4j Knowledge Graph Construction

This stage officially writes the processed data into the Neo4j database and performs graph-level fusion.

#### Import Nodes into Database

```
python step3-0-结点入Neo4j.py
```

**Purpose:** Writes all base node data into Neo4j.
 After running, you can log in to the Neo4j Browser to check if the nodes have been created successfully.

**Tool:** The `DeleteALL.py` script in the same folder can be used to clear the entire database with one click, which is convenient for debugging.

#### Generate Local Fusion Relationships

```
python step3-1-局部融合.py
```

**Purpose:** Computes and generates a list of entity groups that need to be fused locally (`{filename}_group_fusion.jsonl`).

#### Import Local Fusion Relationships into Database

```
python step3-2-局部融合写进Neo4j.py
```

**Purpose:** Writes the fusion relationships generated in the previous step into Neo4j, adding properties to nodes of type `Group`.

#### Calculate Global Fusion Relationships

```
python step3-3-全局融合.py
```

**Purpose:** Performs global fusion calculations across entity groups at the entire graph level, generating the `_group_before_global.jsonl` file.

#### Import Global Fusion Relationships into Database

```
python step3-4-全局融合写進Neo4j.py
```

**Purpose:** Writes the global fusion relationships into Neo4j, completing the final construction of the knowledge graph.

------

### Step 4: Vectorization and Question Answering

This stage creates vector indexes for the knowledge in the graph and starts the question-answering program.

#### Text Vectorization

```
python step-4-1-Embedding.py
```

**Purpose:** Converts the text content of the nodes in the knowledge graph into vector representations.

#### Build Faiss Vector Index

```
python step-4-2-Faiss.py
```

**Purpose:** Uses Faiss to build an efficient local vector database index for fast retrieval.

#### Start Question Answering

```
python QA.py
```

**Configuration:**
 You can modify and configure QA-related parameters and commands in the main function of the `QA.py` file.

------

## 📂 Project Structure

```
.
├── UniRAG/
│   ├── step0-0-文档拆分器.py
│   ├── step0-1-0-文段分析.py
│   ├── step0-1-1-额外转换.py
│   ├── ... (all other py scripts)
│   ├── Neo4j_tools.py
│   └── QA.py
├── your_document.txt
├── requirements.txt
└── README.md
```