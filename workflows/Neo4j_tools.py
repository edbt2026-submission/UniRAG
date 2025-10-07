from neo4j import GraphDatabase
import uuid

from tools import read_jsonl_to_list, read_chunk

# Neo4j 数据库的连接信息
uri = "bolt://localhost:7687"  # 替换为你的 Neo4j 数据库地址
username = "neo4j"  # 替换为你的 Neo4j 用户名
password = "20020515"  # 替换为你的 Neo4j 密码

# 创建 Neo4j 驱动
driver = GraphDatabase.driver(uri, auth=(username, password))


# 插入结点的函数
def create_entity_node(node_id, name, node_type, description, source_index, group_id):
    query = (
        f"CREATE (n:Entity:`{node_type}` {{id: $id, name: $name, description: $description, source_index: $source_index, group_id: $group_id}}) "
        "RETURN n"
    )
    with driver.session() as session:
        result = session.write_transaction(
            lambda tx: tx.run(query, id=node_id, name=name, node_type=node_type, description=description,
                              source_index=source_index, group_id=group_id)
        )
        

def create_relationship_to_chunk(entity_id, source_index):
    query = (
        "MATCH (e:Entity {id: $entity_id}), (c:Chunk {index: $source_index}) "
        "CREATE (e)-[:FROM]->(c) "
        "RETURN e, c"
    )
    with driver.session() as session:
        result = session.write_transaction(
            lambda tx: tx.run(query, entity_id=entity_id, source_index=source_index)
        )


def create_relationships_to_entity(source_id, target_id, relationship_types, description, strength, relation_id):
    with driver.session() as session:
        for relationship_type in relationship_types:
            query = (
                "MATCH (s:Entity {id: $source_id}), (t:Entity {id:$target_id}) "
                f"CREATE (s)-[r:`{relationship_type}` {{description: $description, strength:$strength, id: $relation_id}}]->(t) "
                "RETURN s, t"
            )
            result = session.write_transaction(
                lambda tx, q, sid, tid, desc, strg, rid: tx.run(q, source_id=sid, target_id=tid, description=desc, strength=strg, relation_id=rid),
                query, source_id, target_id, description, strength, relation_id
            )


# 创建单个chunks结点
def create_chunk_node(chunk_uuid, filename, text, index):
    # 定义 Cypher 查询
    query = (
        "CREATE (n:Chunk {id: $id, filename: $filename, text: $text, index: $index}) "
        "RETURN n"
    )
    # 执行插入操作
    with driver.session() as session:
        result = session.write_transaction(
            lambda tx: tx.run(query, id=chunk_uuid, filename=filename, text=text, index=index)
        )


def create_groups():
    # 查询所有不同的group_id
    query = """
        MATCH (e:Entity)
        WITH DISTINCT e.group_id AS group_id
        MERGE (g:Group {group_id: group_id})
        RETURN g
        """

    with driver.session() as session:
        session.write_transaction(lambda tx: tx.run(query))

    # 创建belong关系
    query = """
        MATCH (e:Entity), (g:Group {group_id: e.group_id})
        MERGE (e)-[:BELONGS_TO]->(g)
        """

    with driver.session() as session:
        session.write_transaction(lambda tx: tx.run(query))


def create_global(id, name, desc, group_ids, source_index, other_names):
    other_names = list(set(other_names))
    query = """
    Create (g:Global {global_id: $id, global_name: $name, description: $desc, group_ids: $group_ids, source_index: $source_index, other_names: $other_names })
    return g
    """

    with driver.session() as session:
        session.write_transaction(
            lambda tx: tx.run(query, id=id, name=name, desc=desc, group_ids=group_ids, source_index=source_index, other_names=other_names)
        )



    with driver.session() as session:
        for group_id in group_ids:
            query = """
            MATCH (g:Group {group_id: $group_id}), (g2:Global {global_id: $id})
            MERGE (g)-[:CONTAINS]->(g2)
            """

            session.write_transaction(
                lambda tx: tx.run(query, group_id=group_id, id=id)
            )


def create_is_relationship():
    # 查询所有具有相同 group_id 的 Entity 结点，并在它们之间建立双向的 is 关系
    query = """
    MATCH (e1:Entity), (e2:Entity)
    WHERE e1.group_id = e2.group_id AND e1 <> e2
    MERGE (e1)-[:IS]->(e2)
    MERGE (e2)-[:IS]->(e1)
    """

    with driver.session() as session:
        session.write_transaction(lambda tx: tx.run(query))


def delete_is_relationship():
    # 删除所有 Entity 结点之间的 IS 关系
    query = """
    MATCH (e1:Entity)-[r:IS]->(e2:Entity)
    DELETE r
    """

    with driver.session() as session:
        session.write_transaction(lambda tx: tx.run(query))



def read_ten_nodes(tx):
    # 查询前10个节点
    result = tx.run("MATCH (n) RETURN n LIMIT 10")
    nodes = []
    for record in result:
        # 假设节点有一个属性'name'，根据实际情况调整
        node_name = record["n"]["id"]
        nodes.append(node_name)
    return nodes

def test_neo4j():
    # 读取10个节点
    with driver.session() as session:
        nodes = session.read_transaction(read_ten_nodes)
        for node in nodes:
            print(node)


# 查询所有同一标签的结点
def fetch_all_nodes(node_type):
    query = f"MATCH (n:{node_type}) RETURN n"
    with driver.session() as session:
        result = session.run(query)
        return result.data()  # 返回格式化后的数据


def fetch_nodes(label, key, value):
    query = f"MATCH (n:{label} {{ {key}: '{value}' }}) RETURN n"
    # print(query)
    with driver.session() as session:
        result = session.run(query)
        return result.data()


def update_group_node(group_data):
    query = (
        "MATCH (g:Group {group_id: $group_id}) "
        "SET g.group_name = $group_name, "
        "g.description = $description, "
        "g.group_name_list = $group_name_list, "
        "g.source_index = $source_index "
        "RETURN g"
    )
    with driver.session() as session:
        result = session.write_transaction(
            lambda tx: tx.run(query, group_id=group_data['group_id'], group_name=group_data['group_name'],
                              description=group_data['description'], group_name_list=group_data['group_name_list'],
                              source_index=group_data['source_index'])
        )





def add_embedding_attr(label, key, value, name, desc, hybrid):
    query = (
        f"MATCH (n:{label} {{{key}: $value}}) "
        "SET n.embedding_name = $name, "
        "n.embedding_description = $desc, "
        "n.embedding_hybrid = $hybrid "
        "RETURN n"
    )
    with driver.session() as session:
        result = session.write_transaction(
            lambda tx: tx.run(query, value=value, name=name, desc=desc, hybrid=hybrid)
        )


def add_embedding_attr_global(label, key, value, desc):
    query = (
        f"MATCH (n:{label} {{{key}: $value}}) "
        "SET n.embedding_description = $desc "
        "RETURN n"
    )
    with driver.session() as session:
        result = session.write_transaction(
            lambda tx: tx.run(query, value=value, desc=desc)
        )


def delete_label_all(label):
    query = f"""MATCH (n:`{label}`)
                DETACH DELETE n"""
    with driver.session() as session:
        result = session.write_transaction(lambda tx: tx.run(query))


def get_relations():
    query = """MATCH ()-[r]->()
    WHERE r.id is not null
    RETURN r"""

    with driver.session() as session:
        result = session.run(query)
        relations = [record["r"] for record in result]
    return relations


def add_relation_attr(id, k, v):
    query = f"""MATCH ()-[r]->()
    WHERE r.id = '{id}'
    SET r.{k} = '{v}'
    RETURN r"""

    with driver.session() as session:
        result = session.write_transaction(lambda tx: tx.run(query))


if __name__ == '__main__':
    test_neo4j()
    # delete_label_all("Global")