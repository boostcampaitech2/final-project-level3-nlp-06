# from database import db
from app import db
from models import Validations

# from bm25_test import *

def insert_stocks():
    # method 1 : bm25
    dartdataset, newsdataset = dataloader()
    split_news, dart_dict, bm25, pre_dart, pre_name = make_datasets(dartdataset, newsdataset)
    return_list = get_corp(split_news, dart_dict, bm25, pre_dart, pre_name)
    memo = 'bm25'

    for data in return_list:
        db.session.add(Validations(news=json.dumps(data['news']), keywords=data['keywords'], corporations=data['corporations'], memo=memo))
    db.session.commit()
    print("Insert Validations Finish!")

def update_stocks():
    stocks = Validations.query.get(20)
    news_data = json.loads(stocks.news)
    print(news_data)

    newsdataset = dataloader()

    for idx, data in enumerate(newsdataset['train']):
        if data['id'] == 334905528:
            print(data)
            news = {'id': data['id'], 'original': data['original'], 'data': data['article']}
            stocks.news = json.dumps(news)
            db.session.commit()
            break
    print('Update Validations Finish!')

        
