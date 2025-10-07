# 第一步 准备文件
1.1 把.txt文件放在MyRAG根目录

# 第二步 运行

## 2.1 拆分文档
step0-0-文档拆分器.py

## 2.2 分析文段
step0-1-0-文段分析.py

记得解除analyze_text_large的注释

此处用的是DeepSeek官方进行的分析，官方支持JSON模式

但是可能会服务器拥堵，需要换成Silicon的DeepSeek，但是它不支持JSON

还有就是analyze_text_large这个步骤没有做logger日志，可能执行很长时间没有反应，后面需要修复

## 2.3 检查分析结果
step0-1-1-额外转换.py
结果保存在{filename}_extract_result.jsonl中，初次运行后你可以检查到每个元素的entities是字符串，还没有解析

这一步的用途是检查0-1-0的步骤是否正常生成，正确的json会被转换成dict

注意观察这一步的输出结果

如果显示有错误，则会把抽取结果里的错误结果删掉

重要，这个时候需要重新运行step0-1-0-文段分析.py 自动补上删掉的错误

也就是说0-1-0-文段分析.py和0-1-1-额外转换.py这两个步骤重复循环跑，直到没有错误为止

最后检查{filename}_extract_result.jsonl 里面的条目数少不少，此时素有的entities应该已经是字典了

## 2.4 转换为准备合并的格式
step0-2-结点文件转换.py

结果保存在{filename}_EX.jsonl中

## 2.5 进行局部结点融合
step2_0_go_merge.py

运行时间较长，请耐心等待

结果保存在{filename}_merge.jsonl中

## 2.6 融合检查
排除不合法的融合

step2-1 验证merge合法性.py

生成{filename}_final_merge.jsonl

这一步如果发现了不合法的merge，理论上会自动处理，会重新进行融合，不需要操作

可能等的时间也比较久

## 2.7 创建融合索引，准备入库

step2-2-结点融合.py

生成{filename}_EX_pro.jsonl

# 第三步 Neo4j图谱构建

## 3.1 启动你的Neo4j 
记得修改Neo4j_tools.py中的配置

## 3.2 结点入n4j
step3-0-结点入Neo4j.py

运行完之后可以登录网页检查结点情况

同文件夹下的DeleteALL.py 文件 运行可以一键删库

## 3.3 局部融合（写入本地文件）
step3-1-局部融合.py

生成{filename}_group_fusion.jsonl 需要合并的实体组列表

生成{filename}_group_stay.jsonl 不需要合并的实体组列表（因为这些组里只有一个元素，不需要进行融合）

## 3.4 融合进Neo4j
step3-2-局部融合写进Neo4j.py

给所有的Group结点添加属性

## 3.5 全局融合
step3-3-全局融合.py

产生_group_before_global.jsonl文件

调试时记得注释掉reset_jsonl_file方法

## 3.6 全局融合写进Neo4j
step3-4-全局融合写进Neo4j.py

# 第四步 观察图谱

## 4.1 向量化并存入数据库
step-4-1-Embedding.py

## 4.2 向量保存进本地向量数据库
step-4-2-Faiss.py

使用Faiss构建索引

使用前安装Faiss
```
pip install faiss-cpu
```

# 第四步 开始问答
QA.py

在main方法中可以找到运行的命令

可以使用Pro/Qwen/Qwen2.5-7B-Instruct这种小参数量模型
且在小参数量模型的情况下，图谱回答质量显著高于朴素RAG