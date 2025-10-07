from neo4j import GraphDatabase

# Neo4j数据库连接信息
uri = "bolt://localhost:7687"
user = "neo4j"
password = "20020515"

# 连接到Neo4j数据库
driver = GraphDatabase.driver(uri, auth=(user, password))

def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")

# 清空数据库
with driver.session() as session:
    session.write_transaction(clear_database)

# 关闭数据库连接
driver.close()