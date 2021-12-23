import json
import re

from konlpy.tag import Mecab
from rank_bm25 import BM25Okapi
from krwordrank.word import summarize_with_keywords

from models import Corporations


def make_dart_datasets():
    contents_list = ['회사의 개요',  '사업의 개요', '주요 제품 및 서비스','주요계약 및 연구개발활동']

    dart_dict = {
        'corp_name' : [],
        'corp_text' : [],
    }
    pre_dart = {}
    pre_name = {}

    tokenizer = Mecab()
    tokenized_texts = []

    corporations = Corporations.query.all()
    for corp in corporations:
        data = json.loads(corp.details)
        dart_dict['corp_name'].append(data['corp_name'])
        dart_text = ''
        for content in contents_list:
            dart_text += data[content]
        dart_dict['corp_text'].append(dart_text)
        preprocessed_dart_text = preprocessing(dart_text)
        pre_dart[data['corp_name']] = preprocessed_dart_text
        pre_name[preprocessed_dart_text] = data['corp_name']
        tokenized_texts.append(tokenizer.nouns(preprocessed_dart_text))

    bm25 = BM25Okapi(tokenized_texts)
    
    return dart_dict, pre_dart, pre_name, bm25



def get_corporations(news):
    inside_corps = set()

    news = news.strip()

    dart_dict, pre_dart, pre_name, bm25 = make_dart_datasets()

    stopwords=['모두','지난','있다','회의','강조','성과','오는','모집','과정','계약',"체결","건립",'현대식','사업',
                '주관사로','통해','일까지','이벤트','지원','지역','기자','실험','한다','진행','설명회','예정이다',
                '이번','방문','올해','제공','찾아가','재림','당시','한국','프로젝트','있는','추진','다양한','적극',
                '위해','나갈','협력','협약','상당의','물품','명으로','지난해','증가했다','의한','영향','보면','통계',
                '순위','명당','관련','업체','부실','관심','명칭','합작법인','일자리','시흥배곧','서울대','했다',
                '한다고','위원장은','기반','혁신','인재','마련','정책','권고안','함께','시장','성장','대비','전월',
                '전년','동월','물가','하락','소비자','상승','운영','증가','억원','억만원','할인','사용','것으']

    keywords = summarize_with_keywords([news], min_count=4, max_length=7,
        beta=0.85, max_iter=10,stopwords=stopwords,verbose=True)
    tokenized_query = list(keywords.keys())

    for key in news.split(' '):
        if key in dart_dict['corp_name']:
            inside_corps.add(key)
    
    for key in tokenized_query:
        if key in dart_dict['corp_name']:
            inside_corps.add(key)
            
    inside_corps = list(inside_corps)

    doc_scores = bm25.get_scores(tokenized_query)
    if max(doc_scores) < 3:
        return []
    else:
        list_dart_n = bm25.get_top_n(tokenized_query, list(pre_dart.values()), n=3)
        corporation_names = [pre_name[dart] for dart in list_dart_n]
        return corporation_names
    

def preprocessing(s): 
    hangul = re.compile('[^ ㄱ-ㅣ가-힣+]')
    result = hangul.sub('', s)
    return result

