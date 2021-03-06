import urllib

import json
import datasets
import re

from config import *
from app import db
from models import News, Corporations
from time import sleep

def insert_news():
    corporations = Corporations.query.all()
    for i, corp in enumerate(corporations):
        if i != 0 and i % 10 == 0:
            sleep(1)
        detail = json.loads(corp.details)
        stock_code = str(detail['주식 코드']).zfill(6)
        response = get_news(f'{corp.name} {stock_code}')
        if response['code'] == 200:
            db.session.add(News(id=corp.id, news=response['data'], corporation_id=corp.id))
    db.session.commit()
    print("Insert News Finish!")


def update_news():
    news = News.query.all()
    if len(news) == 0:
        init_news()

    data = Corporations.query.all()
    for i in range(len(data)):
        if i != 0 and i % 10 == 0: # 초당 10회가 넘어가면 에러 발생
            sleep(1)
        detail = json.loads(data[i].details)
        stock_code = str(detail['주식 코드']).zfill(6)
        response = get_news(f'{data[i].name} {stock_code}')
        if response['code'] == 200:
            news_data = News.query.filter(News.corporation_id == data[i].id).one()
            news_data.news = response['data']
    db.session.commit()
    print("Update News Finish!")


def get_news(search_text):
    encText = urllib.parse.quote(search_text)

    url = "https://openapi.naver.com/v1/search/news.json?query=" + encText + "&sort=sim&display=5"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",NAVER_CLIENT_ID)
    request.add_header("X-Naver-Client-Secret",NAVER_CLIENT_SECRET)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        re = json.loads(response_body.decode('utf-8'))
    else:
        return {'code': rescode}
    results = json.dumps(re['items'])
    return {'code': rescode, 'data': results}

def preprocess_news(data):
    data = re.sub('<b>', '', data)
    data = re.sub('</b>', '', data)
    data = re.sub('&[A-Za-z]+;', '', data)
    return data

