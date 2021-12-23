from flask import Blueprint, render_template, request, redirect, url_for
import json
from datetime import datetime
from dateutil import tz

from app import db
from models import Evaluations, Validations, News, Corporations

from stocks import *
from naver_news import *
from corporations import *
from scheduler import *

from bm25_model import *
import kobert_text_classification
from kobert_text_classification import BERTClassifier

import time
from time import sleep

import torch

import gluonnlp as nlp
import numpy as np
from torch import nn

from torch.utils.data import Dataset
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

import ensemble


bp = Blueprint('main', __name__, url_prefix='/')

start_news_update_scheduler()

# ensemble_model = ensemble.Models(num_prediction=20)

@bp.route('/')
def hello():
    ## 처음에 new data 추가할시에만 사용
    # init_data()
    # DBTEST.main()
    return render_template('index.html')

@bp.route('/stock')
def stock():
    ## 시연용
    # korea_timezone =tz.gettz('Asia/Seoul')
    # newsTextData = request.args.get('newsTextData')
    # flag = 1
    # stock_corp = []
    # if '연말까지' in newsTextData:
    #     stock_corp = ['KCC건설', '현대건설', '한진중공업']
    # elif '윤석열 국민의힘 대통령후보가' in newsTextData:
    #     stock_corp = ['에이비온', '앱클론', '압타머사이언스']
    # elif '메신저 리보핵산(mRNA)' in newsTextData:
    #     stock_corp = ['진바이오텍', '중앙백신', '진원생명과학']
    # elif '배우 한예슬이 커피도 럭셔리한' in newsTextData:
    #     stock_corp = ['리노스', '모비릭스', '경인양행']
    # data = Corporations.query.filter(Corporations.name.in_(stock_corp)).all()
    # for i in range(len(data)):
    #     news_data = json.loads(data[i].news_set[0].news)
    #     for j in range(len(news_data)):
    #         news_data[j]['preprocess_title'] = preprocess_news(news_data[j]['title'])
    #         news_data[j]['preprocess_description'] = preprocess_news(news_data[j]['description'])
    #         news_data[j]['pubDate'] = datetime.strptime(news_data[j]['pubDate'].strip(), '%a, %d %b %Y %H:%M:%S %z').replace(tzinfo=korea_timezone).strftime('%Y년 %m월 %d일 %H시 %M분')
    
    #     data[i].news_data = news_data
    # sleep(5)
    


    ## 원래 코드
    start = time.time()
    korea_timezone =tz.gettz('Asia/Seoul')

    newsTextData = request.args.get('newsTextData')
    if newsTextData == None:
        return render_template('index.html', newsTextData=newsTextData, stock_corp=[], flag=0)
    # flag = kobert_text_classification.predict(newsTextData)
    flag = 0
    # print('flag', flag)
    # stock_corp = get_corporations(newsTextData)


    topk = 3
    ensemble_model = ensemble.Models(num_prediction=20)
    # news_keywords : 앙상블 결과로 뉴스에서 뽑은 키워드 -> [str]
    stock_corp, news_keywords = ensemble.ensemble_inference_real_time(ensemble_model, newsTextData, topk)
    print(news_keywords)
    
    
    # stock_corp = ['삼성전자', 'LG전자', 'LG이노텍']
    data = Corporations.query.filter(Corporations.name.in_(stock_corp)).all()
    for i in range(len(data)):
        data[i].detail = json.loads(data[i].details)
        # data[i].news_data = json.loads(data[i].news_set[0].news)
        news_data = json.loads(data[i].news_set[0].news)
        for j in range(len(news_data)):
            news_data[j]['preprocess_title'] = preprocess_news(news_data[j]['title'])
            news_data[j]['preprocess_description'] = preprocess_news(news_data[j]['description'])
            news_data[j]['pubDate'] = datetime.strptime(news_data[j]['pubDate'].strip(), '%a, %d %b %Y %H:%M:%S %z').replace(tzinfo=korea_timezone).strftime('%Y년 %m월 %d일 %H시 %M분')
        data[i].news_data = news_data
    print('걸리는 시간: ', time.time() - start)
    return render_template('index.html', newsTextData=newsTextData, stock_corp=data, flag=flag)

@bp.route('/evaluation')
@bp.route('/evaluation/<int:eval_id>')
def evaluation(eval_id=None):
    nickname = request.args.get('nickname')
    if nickname == None:
        return render_template('evaluation.html', idx=1, nickname=nickname, percent=0, data=None)
    print(eval_id)
    data = Validations.query.filter_by(id=eval_id).first()
    print(data)
    
    value = request.args.get('evaluationRadioOptions')

    if data == None:
        if value == None:
            return render_template('evaluation.html', idx=0, nickname=nickname, percent=0, data=None)
        else:
            value = int(value)
            result = Evaluations.query.get(eval_id-1)
            print(result)
            if result:
                score = json.loads(result.score)
                if type(score) == str:
                    score = json.loads(score)
                score[nickname] = value
                total = 0
                result.count = len(score.keys())
                for key in score:
                    total += score[key]
                result.mean = round((total/result.count), 2)
                result.score = json.dumps(score)
            else:
                score = json.dumps({nickname: value})
                insert_data = Evaluations(id=eval_id-1, count=1, score=score, mean=value, validation_id=eval_id-1)
                db.session.add(insert_data)
            db.session.commit()
            return render_template('evaluation.html', idx=1, nickname=nickname, percent=0, data=None)
    
    data.news_data = json.loads(data.news)
    db_corporations = Corporations.query.filter(Corporations.name.in_(list(data.corporations.split(',')))).all()
    
    corporations = list()
    if len(db_corporations) > 0:
        for i in range(len(db_corporations)):
            corporations.append(json.loads(db_corporations[i].details))
    else:
        for name in list(data.corporations.split(',')):
            corporations.append({'corp_name': name, '회사의 개요': '', '사업의 개요': '', '주요 제품 및 서비스': '', '주요계약 및 연구개발활동': ''})

    data.corporations_data = corporations
    
    if value:
        value = int(value)
        result = Evaluations.query.get(eval_id-1)
        if result:
            score = json.loads(result.score)
            if type(score) == str:
                score = json.loads(score)
            score[nickname] = value
            total = 0
            result.count = len(score.keys())
            for key in score:
                total += score[key]
            result.mean = round((total/result.count), 2)
            result.score = json.dumps(score)
        else:
            score = json.dumps({nickname: value})
            insert_data = Evaluations(id=eval_id-1, count=1, score=score, mean=value, validation_id=eval_id-1)
            db.session.add(insert_data)
        db.session.commit()
    
    percent = round(((eval_id-1)/Validations.query.count()) * 100, 2)
    return render_template('evaluation.html', idx=eval_id, nickname=nickname, percent=percent, data=data)


@bp.route('/result')
@bp.route('/result/<method>')
def result(method=''):
    if method == 'ensemble':
        method = 'bm25, NER, doc2vec'
    data = Validations.query.filter(Validations.memo==method).all()
    members = set()
    results = list()
    for stock in data:
        stock.news_data = json.loads(stock.news)
        if len(stock.eval_set) > 0:
            stock.count = stock.eval_set[0].count
            stock.score = json.loads(stock.eval_set[0].score)
            if type(stock.score) == str:
                stock.score = json.loads(stock.score)
            for member in stock.score.keys():
                members.add(member)
            stock.mean = stock.eval_set[0].mean
        else:
            stock.count = 0
            stock.score = {}
            stock.mean = 0
        results.append(stock)

    return render_template('result.html', method=method, results=results, members=list(members))

def init_data():
    # TODO: stocks에 테스트 데이터 입력 - news : 뉴스데이터, keywords : 키워드 corporations : 관련주명, memo : 어떤 모델 사용했는지
    insert_stocks()
    # TODO: corporations에 기업 정보 입력 - name : 기업명, details : DART 정보, summary : 요약 정보
    insert_corporations()
    # TODO: news에 뉴스데이터 입력 - news : corporation_id에 해당하는 기업에 대한 최신 뉴스 5개
    insert_news()

def insert_data():
    dartdataset, newsdataset = bm25_test.dataloader()
    split_news, dart_dict, bm25, pre_dart, pre_name = bm25_test.make_datasets(dartdataset, newsdataset)
    return_list = bm25_test.print_corp(1, split_news, dart_dict, bm25, pre_dart, pre_name)
    
    for data in return_list:
        stock_value = [d['name'] for d in data['stocks']]
        stock_value = ','.join(stock_value)
        db.session.add(Validations(news=json.dumps(data['news']), stock_value=stock_value))
    db.session.commit()

def update_data():
    dartdataset, newsdataset = bm25_test.dataloader()
    split_news, dart_dict, bm25, pre_dart, pre_name = bm25_test.make_datasets(dartdataset, newsdataset)
    return_list = bm25_test.print_corp(1, split_news, dart_dict, bm25, pre_dart, pre_name)
    
    stocks = Validations.query.all()
    for i in range(len(stocks)):
        stocks[i].news = json.dumps(return_list[i]['news'])
        stocks[i].stock_value = json.dumps(return_list[i]['stocks'])
    
    evaluations = Evaluations.query.all()
    db.session.delete(evaluations)

    # for idx, data in enumerate(return_list):
        # db.session.add(Validations(news=data['news'], stock_value=json.dumps(data['stocks'])))
    db.session.commit()

def confirm_data():
    stock = Validations.query.get(1)
    stock.stock_value = json.loads(stock.stock_value)
    print(stock.stock_value)

def update_eval():
    # evaluations = Evaluations.query.all()
    # idx = 19
    # while idx > 1:
    #     evaluations[idx].mean = evaluations[idx-1].mean
    #     evaluations[idx].score = evaluations[idx-1].score
    #     evaluations[idx].count = evaluations[idx-1].count
    #     idx -= 1
    # if idx == 1:
    #     evaluations[idx].mean = 1
    #     evaluations[idx].count = 1
    #     evaluations[idx].score = json.dumps({"정찬미": 1})
    # db.session.commit()

    evaluation = Evaluations.query.get(2)
    evaluation.score = json.dumps({'정찬미': 1})
    db.session.commit()