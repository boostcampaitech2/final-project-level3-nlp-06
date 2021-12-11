import pandas as pd
import datasets
import re 
from konlpy.tag import Mecab
from rank_bm25 import BM25Okapi
from krwordrank.word import summarize_with_keywords


#data loader
def dataloader() :
    dartdataset = datasets.load_dataset('nlpHakdang/beneficiary',  data_files = "dart_ver1_2.csv", use_auth_token='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')
    newsdataset = datasets.load_dataset('nlpHakdang/aihub-news30k',  data_files = "news_train_1_0.csv", use_auth_token='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')

    return dartdataset, newsdataset

def preprocessing(s): 
        hangul = re.compile('[^ ㄱ-ㅣ가-힣+]')
        result = hangul.sub('', s)
        return result


def make_datasets(dartdataset, newsdataset):
    contents_list = ['회사의 개요',  '사업의 개요', '주요 제품 및 서비스','주요계약 및 연구개발활동']

    dart_dict= {
        "corp_name" :[],
        "corp_text":[]
    }

    for corp in dartdataset['train']:
        dart_text = ""
        dart_dict['corp_name'].append(corp['corp_name'])
        for content in contents_list:
            dart_text += corp[content]
        dart_dict['corp_text'].append(dart_text)

    pre_dart = {}
    for idx in range(len(dart_dict['corp_name'])):
        pre_dart[dart_dict['corp_name'][idx]] = preprocessing(dart_dict['corp_text'][idx])
    pre_name = {}
    for idx in range(len(dart_dict['corp_name'])):
        pre_name[preprocessing(dart_dict['corp_text'][idx])] = dart_dict['corp_name'][idx]

    news_list = []
    for news in tqdm(newsdataset['train']):
        if news['category'] == '경제' or news['category'] =='기업':
            news_list.append(news['article'])

    split_news=[]
    for article in news_list:
        split_news.append(article.split("."))


    tokenizer = Mecab()
    tot = []
    for txt in tqdm(list(pre_dart.values())):
        tot.append(tokenizer.nouns(txt))

    bm25 = BM25Okapi(tot)

    return split_news, dart_dict, bm25, pre_dart, pre_name


def print_corp(newsindex, split_news, dart_dict, bm25, pre_dart, pre_name):
    target_idx=newsindex
    target = split_news[target_idx]
    inside_corp = []

    stopwords=['모두','지난','있다','회의','강조','성과','오는','모집','과정','계약',"체결","건립",'현대식','사업',
            '주관사로','통해','일까지','이벤트','지원','지역','기자','실험','한다','진행','설명회','예정이다',
            '이번','방문','올해','제공','찾아가','재림','당시','한국','프로젝트','있는','추진','다양한','적극',
            '위해','나갈','협력','협약','상당의','물품','명으로','지난해','증가했다','의한','영향','보면','통계',
            '순위','명당','관련','업체','부실','관심','명칭','합작법인','일자리','시흥배곧','서울대','했다',
            '한다고','위원장은','기반','혁신','인재','마련','정책','권고안','함께','시장','성장','대비','전월',
            '전년','동월','물가','하락','소비자','상승','운영','증가','억원','억만원','할인','사용','것으']
    keywords = summarize_with_keywords(target, min_count=4, max_length=7,
        beta=0.85, max_iter=10,stopwords=stopwords,verbose=True)
    tokenized_query = list(keywords.keys())

    for key in ' '.join(target).split(" "):
        if key in dart_dict['corp_name']:
            inside_corp.append(key)

    for key in tokenized_query:
        if key in dart_dict['corp_name']:
            inside_corp.append(key)
    
    inside_corp = list(set(inside_corp))
    
    doc_scores = bm25.get_scores(tokenized_query)
    if max(doc_scores) <3:
        #마땅한 회사가 없음
        return False
    else:
        list_dart_n = bm25.get_top_n(tokenized_query, list(pre_dart.values()), n=3)
        corps = [pre_name[dart] for dart in list_dart_n]
    """
    tokenized_query : keywords
    inside_corp : 언급된 회사
    " ".join(target) : 뉴스텍스트
    corps : topk의 회사이름
    list_dart_n : topk dart
    """
        return tokenized_query, inside_corp, " ".join(target),corps, list_dart_n