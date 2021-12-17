from flask import Blueprint, render_template, request, redirect, url_for
import json

from app import db
from models import Evaluations, Stocks, News, Corporations

from stocks import *
from naver_news import *
from corporations import *

import DBTEST

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def hello():
    ## 처음에 new data 추가할시에만 사용
    # init_data()

    ## Update naver news 
    # update_news()
    return render_template('index.html')

@bp.route('/stock')
def stock():
    newsTextData = request.args.get('newsTextData')
    if newsTextData == None:
        return render_template('index.html', newsTextData=newsTextData, stock_corp=[])
    
    stock_corp = ['삼성전자', 'LG전자', 'LG이노텍']
    data = Corporations.query.filter(Corporations.name.in_(stock_corp)).all()
    for i in range(len(data)):
        data[i].detail = json.loads(data[i].details)
        news_data = json.loads(data[i].news_set[0].news)
        for j in range(len(news_data)):
            news_data[j]['preprocess_title'] = preprocess_news(news_data[j]['title'])
            news_data[j]['preprocess_description'] = preprocess_news(news_data[j]['description'])
        data[i].news_data = news_data
    
    return render_template('index.html', newsTextData=newsTextData, stock_corp=data)

@bp.route('/evaluation')
@bp.route('/evaluation/<int:eval_id>')
def evaluation(eval_id=None):
    data = Stocks.query.all()
    nickname = request.args.get('nickname')
    if nickname == None:
        return render_template('evaluation.html', idx=1, nickname=nickname, percent=0, data=None)
    
    if len(data) == 0:
        return render_template('evaluation.html', idx=1, nickname=nickname, percent=0, data=None)

    value = request.args.get('evaluationRadioOptions')
    if value == None and eval_id == None:
        return render_template('evaluation.html', idx=1, nickname=nickname, percent=0, data=None)
    
    if eval_id == 1 and value == None:
        data[0].news_data = json.loads(data[0].news)
        stocks = Corporations.query.filter(Corporations.name.in_(list(data[0].corporations.split(',')))).all()
        for i in range(len(stocks)):
            stocks[i].detail = json.loads(stocks[i].details)
        data[0].stocks = stocks
        return render_template('evaluation.html', idx=1, nickname=nickname, percent=0, data=data[0])        
    elif value != None:
        if eval_id == None:
            eval_id = 21
        value = int(value)
        result = Evaluations.query.get(eval_id-1)
        if result:
            score = json.loads(result.score)
            score[nickname] = value
            total = 0
            result.count = len(score.keys())
            for key in score:
                total += score[key]
            result.mean = round((total/(result.count)), 2)
            result.score = json.dumps(score)
        else:
            score = json.dumps({nickname : value})
            insert_data = Evaluations(id=eval_id-1, count=1, score=score, mean=value, stock_id=eval_id-1)
            db.session.add(insert_data)
        db.session.commit()

    idx = eval_id-1
    percent = round(((eval_id-1)/len(data)) * 100, 2)
    if len(data) == eval_id-1:
        return render_template('evaluation.html', idx=eval_id, nickname=nickname, percent=percent, data=None)

    data[idx].news_data = json.loads(data[idx].news)
    stocks = Corporations.query.filter(Corporations.name.in_(list(data[idx].corporations.split(',')))).all()
    for i in range(len(stocks)):
        stocks[i].detail = json.loads(stocks[i].details)
    data[idx].stocks = stocks
    return render_template('evaluation.html', idx=eval_id, nickname=nickname, percent=percent, data=data[idx])


@bp.route('/result')
@bp.route('/result/<method>')
def result(method=''):
    data = Stocks.query.filter(Stocks.memo==method).all()
    members = set()
    results = list()
    for stock in data:
        stock.news_data = json.loads(stock.news)
        if len(stock.eval_set) > 0:
            stock.count = stock.eval_set[0].count
            stock.score = json.loads(stock.eval_set[0].score)
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