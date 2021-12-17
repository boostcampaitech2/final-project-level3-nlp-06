from app import db
from models import Stocks

from bm25_test import *

def insert_stocks():
    # method 1 : bm25
    dartdataset, newsdataset = dataloader()
    split_news, dart_dict, bm25, pre_dart, pre_name = make_datasets(dartdataset, newsdataset)
    return_list = get_corp(split_news, dart_dict, bm25, pre_dart, pre_name)
    memo = 'bm25'

    for data in return_list:
        db.session.add(Stocks(news=json.dumps(data['news']), keywords=data['keywords'], corporations=data['corporations'], memo=memo))
    db.session.commit()
    print("Insert Stocks Finish!")