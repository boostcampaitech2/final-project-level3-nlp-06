from datasets import Dataset, load_dataset
import pandas as pd
import os 
dataset = load_dataset("nlpHakdang/beneficiary", data_files = "dart_ver1_2.csv", use_auth_token= "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")
df = pd.DataFrame(dataset['train'])
for i in range(len(df)) :    
    data = df.iloc[i]
    print("=============================={}==============================".format(data['corp_name']))
    print(data["회사의 개요"])    
    input()    
    print("=============================={}==============================".format(data['corp_name']))
    print(data["주요 제품 및 서비스"])    
    input()    
    os.system('clear')


# import pickle
# from bs4 import BeautifulSoup
# import pandas
# with open('data/dart/corp_2500_result_without_table.pickle', "rb") as fp:
#   data = pickle.load(fp)
# soup = BeautifulSoup(data[0]['original_xml'], 'lxml')
# tables = soup.find_all('table')
# for table in tables : 
#     trs = table.find_all('tr')
#     for i in trs :
#         tds = i.find_all('td')
#         print("확인")