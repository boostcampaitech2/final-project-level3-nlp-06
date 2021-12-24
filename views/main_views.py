from flask import Blueprint, render_template, request, redirect, url_for
import json
from datetime import datetime
from dateutil import tz
import os

from app import db
from models import Evaluations, Validations, News, Corporations

from stocks import *
from naver_news import *
from corporations import *
from scheduler import *

import kobert_text_classification
from kobert_text_classification import BERTClassifier, BERTModels

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
# from naver_news import update_news
from model import neo4j_graph

from stock_visualier import visualize

bp = Blueprint('main', __name__, url_prefix='/')
ensemble_model = ensemble.Models(num_prediction=20)
classification_model = kobert_text_classification.BERTModels()


start_news_update_scheduler()

@bp.route('/')
def hello():
    return render_template('index.html')

@bp.route('/stock')
def stock():
    start = time.time()
    korea_timezone =tz.gettz('Asia/Seoul')

    # 뉴스텍스트데이터가 정상적으로 들어오는지 확인
    newsTextData = request.args.get('newsTextData')
    if newsTextData == None:
        return render_template('index.html', keyword_graphs=[], newsTextData=newsTextData, stock_corp=[], inside_corp=[], flag=2)
    elif len(newsTextData) < 50:
        return render_template('index.html', keyword_graphs=[], newsTextData=newsTextData, stock_corp=[], inside_corp=[], flag=2)
    flag = classification_model.predict(newsTextData)

    topk = 3
    stock_corp, keywords, inside_corp = ensemble.ensemble_inference_real_time(ensemble_model, newsTextData, topk)
    
    # 키워드 관련 기업
    keyword_graphs = list()
    for keyword in keywords:
        if f'{keyword}.png' in os.listdir('/opt/ml/coding/static/img/neo4j/'):
            keyword_graphs.append({'keyword': keyword, 'img_url': f'img/neo4j/{keyword}.png'})
    
    # 관련 주식
    stock_corps = Corporations.query.filter(Corporations.name.in_(stock_corp)).all()
    for i in range(len(stock_corps)):
        stock_corps[i].detail = json.loads(stock_corps[i].details)
        stock_corps[i].summary = json.loads(stock_corps[i].summaries)
        news_data = json.loads(stock_corps[i].news_set[0].news)
        for j in range(len(news_data)):
            news_data[j]['preprocess_title'] = preprocess_news(news_data[j]['title'])
            news_data[j]['preprocess_description'] = preprocess_news(news_data[j]['description'])
            news_data[j]['pubDate'] = datetime.strptime(news_data[j]['pubDate'].strip(), '%a, %d %b %Y %H:%M:%S %z').replace(tzinfo=korea_timezone).strftime('%Y년 %m월 %d일 %H시 %M분')
        stock_corps[i].news_data = news_data
        stock_code = str(stock_corps[i].detail['주식 코드']).zfill(6)
        start_date = datetime(2021, 12, 1)
        end_date = datetime(2021, 12, 31)
        img_url = visualize(start_date, end_date, stock_code)
        stock_corps[i].img_url = img_url

    # 뉴스에 언급된 기업
    inside_corps = Corporations.query.filter(Corporations.name.in_(inside_corp)).all()
    for i in range(len(inside_corps)):
        inside_corps[i].detail = json.loads(inside_corps[i].details)
        inside_corps[i].summary = json.loads(inside_corps[i].summaries)
        news_data = json.loads(inside_corps[i].news_set[0].news)
        for j in range(len(news_data)):
            news_data[j]['preprocess_title'] = preprocess_news(news_data[j]['title'])
            news_data[j]['preprocess_description'] = preprocess_news(news_data[j]['description'])
            news_data[j]['pubDate'] = datetime.strptime(news_data[j]['pubDate'].strip(), '%a, %d %b %Y %H:%M:%S %z').replace(tzinfo=korea_timezone).strftime('%Y년 %m월 %d일 %H시 %M분')
        inside_corps[i].news_data = news_data
        stock_code = str(inside_corps[i].detail['주식 코드']).zfill(6)
        start_date = datetime(2021, 12, 1)
        end_date = datetime(2021, 12, 31)
        img_url = visualize(start_date, end_date, stock_code)
        inside_corps[i].img_url = img_url


    print('걸리는 시간: ', time.time() - start)
    return render_template('index.html', keyword_graphs=keyword_graphs, newsTextData=newsTextData, stock_corp=stock_corps, inside_corp=inside_corps, flag=flag)

@bp.route('/evaluation')
@bp.route('/evaluation/<int:eval_id>')
def evaluation(eval_id=None):
    nickname = request.args.get('nickname')
    if nickname == None:
        return render_template('evaluation.html', idx=1, nickname=nickname, percent=0, data=None)
    
    data = Validations.query.filter_by(id=eval_id).first()
    
    value = request.args.get('evaluationRadioOptions')

    if data == None:
        if value == None:
            return render_template('evaluation.html', idx=0, nickname=nickname, percent=0, data=None)
        else:
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
            return render_template('evaluation.html', idx=1, nickname=nickname, percent=0, data=None)
    
    data.news_data = json.loads(data.news)
    db_corporations = Corporations.query.filter(Corporations.name.in_(list(data.corporations.split(',')))).all()
    
    # 기업 정보
    corporations = list()
    for i in range(len(db_corporations)):
        corporations.append(json.loads(db_corporations[i].details))

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