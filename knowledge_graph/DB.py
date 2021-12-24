from neo4j import GraphDatabase
import re
from ast import literal_eval
import os
from numpy.lib.function_base import delete
import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothicOTF')

""" make node & relationship"""
def add_corp(tx, name, corp_code, stock_code, date, report_idx, keyword):
    tx.run("MERGE (c:Corp {name: $name , corp_code : $corp_code, stock_code:$stock_code, date: $date, report_idx: $report_idx, keyword: $keyword})",
           name=name, corp_code=corp_code, stock_code=stock_code, date=date,  report_idx=report_idx, keyword=keyword)

# def add_product(tx): #기업 기준으로 키워드들을 묶는 작업
#     tx.run("MATCH (c:Corp) "
#            "UNWIND c.keyword as k "
#            "MERGE (b:Keyword {name:k}) "
#            "MERGE (c)-[r:Product]->(b)")

def add_product(tx, relation): #기업 기준으로 키워드들을 묶는 작업
    tx.run("MATCH (c:Corp) "           
           "UNWIND c.keyword as k "
           "MERGE (b:Keyword {name:k}) "
           "MERGE (c)-[r:Product {name: $relation}]->(b)", relation=relation) #가능한지 확인해볼것

def add_key(tx, name, relation='', emotion='', corp_list=[], key_list=[]):
    tx.run("MERGE (k:Key {name:$name, relation:$relation, emotion:$emotion, corp_list:$corp_list, key_list:$key_list})", name=name, relation=relation, emotion=emotion, corp_list=corp_list, key_list=key_list)

def add_k2c(tx):    
    tx.run("MATCH (k:Key {name:k.name}) "                      
           "WHERE k.relation='theme' "
           "UNWIND k.corp_list as c "                              
           "MERGE (b:Corp {name:c}) "             
           "MERGE (k)-[r1:Theme {name: k.relation}]->(b) "          
           )
    tx.run("MATCH (k:Key {name:k.name}) "                      
           "WHERE k.relation='product' "        
           "UNWIND k.corp_list as c "                      
           "MERGE (b:Corp {name:c}) "             
           "MERGE (k)-[r1:Product {name: k.relation}]->(b) "          
           )
def add_k2k(tx):  
    tx.run("MATCH (k:Key {name:k.name}) "                      
           "WHERE k.emotion='positive' "        
           "UNWIND k.key_list as c "                      
           "MERGE (b:Key {name:c}) "             
           "MERGE (k)-[r:Positive {name: k.emotion}]->(b) "          
           "MERGE (b)-[r1:Positive {name: k.emotion}]->(k) " 
           )
    tx.run("MATCH (k:Key {name:k.name}) "                      
           "WHERE k.emotion='negative' "        
           "UNWIND k.key_list as c "                      
           "MERGE (b:Key {name:c}) "             
           "MERGE (k)-[r:Negative {name: k.emotion}]->(b) "          
           "MERGE (b)-[r1:Negative {name: k.emotion}]->(k) "          
           )
    tx.run("MATCH (n:Key) "                      
           "WITH n.name as name, COLLECT(n) AS ns "        
           "WHERE size(ns) > 1 "                      
           "CALL apoc.refactor.mergeNodes(ns) YIELD node "
           "RETURN node"
           ) 

""" 한자와 공백 제거 """
# Neo4j -> Gephi 에서 parsing error의 원인이 될 수 있음
def clean_text_for_neo4j(row, column):
    text = row[column]
    try :
        text_list = eval(text)
        answer = [] 
        for text in text_list : 
            text = re.sub(pattern='[^a-zA-Z0-9ㄱ-ㅣ가-힣]', repl='', string=text)
            answer.append(text) 
        # print("영어, 숫자, 한글만 포함 : ", text )
        return answer
    except:
        return [] 



""" 입력 """
# Cyper code를 이용,  크롤링한 Data를 DB에 입력

class neo4jDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))  
        
    def close(self):
        self.driver.close()    

    def create_k2c_graph(self):
        df = pd.read_csv("./flaskr/k2c.csv", encoding='utf8')
        df['corp'] = df.apply(clean_text_for_neo4j, column='corp', axis=1)        
        with self.driver.session() as session:
            """ make node """
            for idx in range(len(df)):        
                session.write_transaction(add_key, name=df.iloc[idx]['name'], relation=df.iloc[idx]['relation'],corp_list=df.iloc[idx]['corp'])    
            session.write_transaction(add_k2c)
   

    def create_k2k_graph(self):
        df = pd.read_csv("./flaskr/k2k.csv", encoding='utf8')
        df['keyword'] = df.apply(clean_text_for_neo4j, column='keyword', axis=1)        
        with self.driver.session() as session:
            for idx in range(len(df)):        
                session.write_transaction(add_key, name=df.iloc[idx]['name'], emotion=df.iloc[idx]['emotion'],key_list=df.iloc[idx]['keyword'])    
            session.write_transaction(add_k2k)
            
    def create_graph(self) :
        df = pd.read_csv("./flaskr/filterd_data_v1.csv", encoding='CP949')
        df['main_task'] = df.apply(clean_text_for_neo4j, axis=1)
        with self.driver.session() as session:
            """ make node """
            for idx in range(len(df)):        
                session.write_transaction(add_corp, name=df.iloc[idx]['corp_name'], corp_code=str(df.iloc[idx]['corp_code']), stock_code=str(df.iloc[idx]['stock_code']), 
                date=str(df.iloc[idx]['date']),  report_idx=str(df.iloc[idx]['report_id']), keyword=df.iloc[idx]['main_task'])    
            session.write_transaction(add_product, "Product")
    

#==============================================삭제==============================================#
    def delete_all(self):
        with self.driver.session() as session:  
             session.write_transaction(self._delete_relation)    
    @staticmethod
    def _delete_relation(tx):
        tx.run("match (a) -[r] -> () delete a, r")
        tx.run("match (a) delete a")
#==============================================테스트==============================================#
    def print_product(self):
        with self.driver.session() as session:
            return session.write_transaction(self._return_product_relation)          
    @staticmethod
    def _return_product_relation(tx):
        result = tx.run("MATCH p=(n1)-[r:Product]->(n2)"
                        " RETURN n1, r, n2"
                        " LIMIT 5")
        DG = nx.DiGraph()
        nodes = ['n1', 'n2'] 
        relations = ['r']                
        corp_node_id = [] 
        keyword_node_id = [] 
        for i, path in enumerate(result): 
            node_dicts=[]        
            for node in nodes :  
                temp_dict = {
                    'name':path[node]['name'],
                    'id': path[node].id, 
                    'labels':path[node].labels, 
                    'properties':dict(path[node])                
                    }   
                if "corp_code" in temp_dict["properties"] :
                    corp_node_id.append(temp_dict['name'])
                else :
                    keyword_node_id.append(temp_dict['name'])
                node_dicts.append(temp_dict)          
                if temp_dict['name'] not in DG : 
                    DG.add_nodes_from([
                    (temp_dict['name'], temp_dict)
                    ])
            for idx, relation in enumerate(relations):            
                e_dict = {
                    'name': "Product",
                    'id':path[relation].id, 
                    'type':path[relation].type, 
                    'properties':dict(path[relation])
                }
                DG.add_edges_from([
                    (node_dicts[idx]['name'], node_dicts[idx+1]['name'], e_dict)
                ])
        return DG, list(set(corp_node_id)), list(set(keyword_node_id))
#==============================================기업과 기업사이의 연관관계 ==============================================#
    def corp2corp(self, corp1, corp2):
        with self.driver.session() as session:
            return session.write_transaction(self._return_corp2corp, corp1, corp2)          
    @staticmethod
    def _return_corp2corp(tx, corp1, corp2):
        result = tx.run(query="MATCH p=(n1)-[r1:Product]-(n2)-[r2:Product]-(n3) "
                        "WHERE n1.name=$corp1 and n3.name=$corp2 "
                        "RETURN n1, n2, n3 ", corp1=corp1, corp2=corp2)
        DG = nx.DiGraph()
        for i, path in enumerate(result):            
            n1_dict = {
                'name':path['n1']['name'],
                'id': path['n1'].id, 
                'labels':path['n1'].labels, 
                'properties':dict(path['n1'])                
            }
            n2_dict = {
                'name':path['n2']['name'],
                'id': path['n2'].id, 
                'labels':path['n2'].labels, 
                'properties':dict(path['n2'])
            }            
            n3_dict = {
                'name':path['n3']['name'],
                'id': path['n3'].id, 
                'labels':path['n3'].labels, 
                'properties':dict(path['n3'])
            }
            # 마찬가지로, edge의 경우도 아래와 같이 정보를 저장한다.
            r1_dict = {
                'name': "주요사업",
                'id':path['r1'].id, 
                'type':path['r1'].type, 
                'properties':dict(path['r1'])
            }
            r2_dict = {
                'name': "주요사업",
                'id':path['r2'].id, 
                'type':path['r2'].type, 
                'properties':dict(path['r2'])
            }
            # print(e_dict)
            # 해당 노드를 넣은 적이 없으면 넣는다.
            if n1_dict['name'] not in DG:
                DG.add_nodes_from([
                    (n1_dict['name'], n1_dict)
                ])
            # 해당 노드를 넣은 적이 없으면 넣는다.
            if n2_dict['name'] not in DG:
                DG.add_nodes_from([
                    (n2_dict['name'], n2_dict)
                ])
            if n3_dict['name'] not in DG:
                DG.add_nodes_from([
                    (n3_dict['name'], n2_dict)
                ])
            # edge를 넣어준다. 노드의 경우 중복으로 들어갈 수 있으므로 중복을 체크해서 넣어주지만, 
            # edge의 경우 중복을 체크하지 않아도 문제없다.
            DG.add_edges_from([
                (n1_dict['id'], n2_dict['id'], r1_dict)
            ])
            DG.add_edges_from([
                (n2_dict['id'], n3_dict['id'], r2_dict)
            ])
        return DG

#==============================================키와 기업사이의 연관관계==============================================#
    def key2corp(self, key, relation):
         with self.driver.session() as session:
            return session.write_transaction(self._return_key2corp, key, relation)          
    @staticmethod
    def _return_key2corp(tx, key, relation):
        
        query="MATCH (n0)<-[r0]-(n1:Key)-[r1]-(n2:Key)-[r2]->(n3) WHERE n1.name={0} and r1.name={1} RETURN n0, r0, n1, n2, r1, r2, n3" #됨
        result = tx.run(query.format("\'"+key+"\'", "\'"+relation+"\'"))
        
        DG = nx.DiGraph()
        nodes = ['n0', 'n1', 'n2', 'n3' ] 
        relations = ['r0', 'r1', 'r2']                
        corp_node_id = [] 
        keyword_node_id = [] 
        for i, path in enumerate(result): 
            node_dicts=[]        
            for node in nodes :  
                temp_dict = {
                    'name':path[node]['name'],
                    'id': path[node].id, 
                    'labels':path[node].labels, 
                    'properties':dict(path[node])                
                    }   
                if "emotion" in temp_dict["properties"] :
                    corp_node_id.append(temp_dict['name'])
                else :
                    keyword_node_id.append(temp_dict['name'])
                node_dicts.append(temp_dict)          
                if temp_dict['name'] not in DG : 
                    DG.add_nodes_from([
                    (temp_dict['name'], temp_dict)
                    ])
            for idx, relation in enumerate(relations):            
                e_dict = {
                    'name': dict(path[relation])['name'],
                    'id':path[relation].id, 
                    'type':path[relation].type, 
                    'properties':dict(path[relation])
                }
                DG.add_edges_from([
                    (node_dicts[idx]['name'], node_dicts[idx+1]['name'], e_dict)
                ])
        return DG, list(set(corp_node_id)), list(set(keyword_node_id))

def draw_graph(DG,  corp_node_id, keyword_node_id, pic_name="fig1.png"):        
    options = {"node_size": 1000, "alpha": 0.9}
    pos = nx.spring_layout(DG, k=1.15)        

    nx.draw(DG, pos=pos, font_family='NanumBarunGothicOTF', font_size=10, with_labels=True, **options)
    #####Edge
    labels = nx.get_edge_attributes(DG, 'name')
    nx.draw_networkx_edge_labels(DG, pos, font_family='NanumBarunGothicOTF', font_size=10, edge_labels=labels)

    #####node
    nx.draw_networkx_nodes(DG, pos, nodelist=corp_node_id, node_color="tab:blue", **options)    
    nx.draw_networkx_nodes(DG, pos, nodelist=keyword_node_id, node_color="tab:red", **options)         
    
    
    plt.savefig('./'+pic_name, dpi=300)



def get_DG():
    greeter = neo4jDB("bolt://localhost:7687", "neo4j", "password")
    DG,  corp_node_id, keyword_node_id = greeter.print_product()    
    draw_graph(DG,  corp_node_id, keyword_node_id)      
    greeter.close()

def get_corp2corp(corp1, corp2):
    greeter = neo4jDB("bolt://localhost:7687", "neo4j", "password")
    DG = greeter.corp2corp(corp1, corp2)    
    pic_name = corp1+corp2+'.png'
    draw_graph(DG, pic_name)
    greeter.close()
    return pic_name

def get_key2corp(key, relation): #relation필요함
    greeter = neo4jDB("bolt://localhost:7687", "neo4j", "password")
    DG,  corp_node_id, keyword_node_id = greeter.key2corp(key, relation)         
    pic_name = key+relation+'.png'
    draw_graph(DG,  corp_node_id, keyword_node_id, pic_name=pic_name)     
    
    greeter.close()
    return pic_name

def delete_all():
    greeter = neo4jDB("bolt://localhost:7687", "neo4j", "password")
    greeter.delete_all()

def create_Product_relation():
    greeter = neo4jDB("bolt://localhost:7687", "neo4j", "password")
    greeter.create_graph()

def create_K2C_relation():
    greeter = neo4jDB("bolt://localhost:7687", "neo4j", "password")
    greeter.create_k2c_graph()

def create_K2K_relation():
    greeter = neo4jDB("bolt://localhost:7687", "neo4j", "password")
    greeter.create_k2k_graph()


delete_all()
create_K2C_relation()
create_K2K_relation()
get_key2corp("전기차", "positive")



