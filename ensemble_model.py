import re
import datasets
from konlpy.tag import Mecab
from rank_bm25 import BM25Okapi
from krwordrank.word import summarize_with_keywords

from model import doc2vec_model
from KoBERT import bm25_ner_serving

def dataloader():
    dartdataset = datasets.load_dataset('nlpHakdang/beneficiary',  data_files = "dart_ver1_2.csv", use_auth_token='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')

    return dartdataset

def make_dart_datasets(corporations):
    contents_list = ['회사의 개요',  '사업의 개요', '주요 제품 및 서비스','주요계약 및 연구개발활동']

    dart_dict = {
        'corp_name' : [],
        'corp_text' : [],
    }
    pre_dart = {}
    pre_name = {}

    tokenizer = Mecab()
    tokenized_texts = []

    # corporations = Corporations.query.all()
    for data in corporations:
        # data = json.loads(corp.details)
        dart_dict['corp_name'].append(data['corp_name'])
        dart_text = ''
        for content in contents_list:
            dart_text += data[content]
        dart_dict['corp_text'].append(dart_text)
        preprocessed_dart_text = orig_preprocessing(dart_text)
        pre_dart[data['corp_name']] = preprocessed_dart_text
        pre_name[preprocessed_dart_text] = data['corp_name']
        tokenized_texts.append(tokenizer.nouns(preprocessed_dart_text))

    bm25 = BM25Okapi(tokenized_texts)
    
    return dart_dict, pre_dart, pre_name, bm25

    


def get_corporations_keyword_bm25(news, dart_dict, pre_dart, pre_name, bm25, num_prediction = 10):
    inside_corps = set()

    news = news.strip()


    

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
        list_dart_n = bm25.get_top_n(tokenized_query, list(pre_dart.values()), n=num_prediction)
        corporation_names = [pre_name[dart] for dart in list_dart_n]
        return corporation_names, keywords
    

def orig_preprocessing(s): 
    hangul = re.compile('[^ ㄱ-ㅣ가-힣+]')
    result = hangul.sub('', s)  
    return result

# input : 뉴스텍스트데이터(string)
# get_corporations(newsTextData)
# output : ['삼성전자', 'LG전자', 'LG이노텍']


re_pattern = r"[가-힣]{0,4} (기자|선임기자|수습기자|특파원|객원기자|논설고문|통신원|연구소장)" # 이름 + 기자

def preprocess(content):
    print(type(content))
    import re
    content = re.sub("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-]+)", "", content)
    content = re.sub("[-=+,#/\?:^@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·▲△▶■]", " ", content)
    content = re.sub(re_pattern, "", content).strip()
    content = re.sub("(.kr)","", content)
    content = ' '.join(content.split())
    return content



if __name__ == '__main__':

    # disable huggingface tokenizer multiprocess warning
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import json

    # # 1. 다트데이터셋 전처리
    dart_dict, pre_dart, pre_name = bm25_ner_serving.get_preprocess_data()

    corp_list = list( dart_dict.keys() )

    # # 2. bm25 모델 로드
    mecab_tokenizer, bm25 = bm25_ner_serving.load_bm25_model(pre_dart)

    # # 3. NER 모델 로드
    pred_config, args, tokenizer, pad_token_label_id, device, model, label_lst = bm25_ner_serving.load_NER()

    news_text = """올해 공모주 시장이 마무리되면서 내년 기업공개(IPO) 대어들에 대한 관심이 높아지고 있다. LG에너지솔루션과 현대엔지니어링이 잇따라 1~2월 일반 청약 일정을 내놓았는데 LG엔솔 청약 증거금을 환불 받은 뒤 다시 현대엔지니어링 청약에 참여할 수 있는 일정이다. 눈에 띄는 점은 두 회사 모두 KB증권이 대표 주관사로 이름을 올리고 있다는 점으로, 일반 투자자 대상 청약 물량이 가장 여유있는 증권사라는 의미다.19일 업계에 따르면 두 IPO 대어가 잇따라 공모 시장에 모습을 드러내면서 투자자들의 기대도 커지고 있다. 특히 LG엔솔 이후 약간의 시간을 두고 현대엔지니어링이 공모에 나서기 때문에 LG엔솔 청약 자금을 다시 현대엔지니어링에 투입할 수 있다. LG엔솔은 내년 1월 18~19일 일반 청약, 21일 증거금 환불, 27일 코스피 입성을 목표로 IPO 일정을 진행하고 있는데 현대엔지니어링은 2월 3~4일 일반 청약에 나선다. LG엔솔의 환불 증거금은 물론, 배정 받은 주식으로 수익을 낸 뒤 그 자금을 다시 한번 현대엔지니어링 청약에 투입할 수 있는 셈이다.LG엔솔과 현대엔지니어링 청약에 모두 참여할 계획인 투자자는 KB증권 창구를 고려해 볼만하다. KB증권이 두 회사의 대표 주관사로 모두 이름을 올리고 있기 때문이다. LG엔솔은 일반 청약자들에 공모가 상단 기준 3조 1,875억 원(1,062만 5,000주)을 조달할 계획인데 이 중 1조 4,609억 원(486만 9,792주)이 KB증권의 몫이다. 공동 주관사인 대신증권과 신한금융투자가 각각 7,305억 원(243만 4,896주)을 일반 투자자들에 내놓는 것과 비교해 두 배 수준으로 규모가 크다.특히 공모주 청약에 고액을 배팅하려는 자산가들은 KB증권에서 486억 원(16만 2,000주) 어치의 청약에 나설 수 있다. 우대 조건에 해당 되는 고객은 1,458억 원(48만 6,000주)까지도 청약이 가능하다. 공동 주관사인 대신증권에서는 360억 원(12만 주), 신한금융투자에서는 243억 원(8만 1,000주)까지 청약 할 수 있는 것에 비해 고액 자산가들에 유리하다. 다만 증권사별 청약 경쟁률, 청약 건수에 따라 최종 배정 공모주 몫이 결정되기 때문에 청약 당일(1월 18~19일) 공모 현황을 최종 확인한 뒤 청약해야 공모주를 한 주라도 더 받을 수 있다."""    
    
    print("load done!")

    infer_dict = {}
    file1 = open('queries_show.txt', 'r')
    Lines = file1.readlines()
    idx = -1
    for line in Lines:
    # while True:
        # news_text = input("input news text\n")

        if line == "\n":
            continue
        news_text = line.strip()
        idx += 1
        infer_dict[idx] = {}
        infer_dict[idx]['news_text'] = news_text

        orig_news_text = news_text
        news_text = preprocess(news_text)

        from time import time
        
        start = time()
        ## 매번 실행헤서 결과를 얻는 부분
        # 4. 입력받은 뉴스데이터
        # news_text = "‘넥슨’ 이름을 단 시가총액 1조원이 넘는 개발사가 조만간 모습을 드러낼 전망이다. 넥슨은 최근 개발 자회사 넷게임즈와 넥슨지티를 합병한다고 밝혔다. 모바일게임과 PC 온라인게임에 각각 강점을 가지고 있는 두 회사의 합병을 통해 시너지 효과를 낼 것으로 보인다. 아울러 넥슨의 국내 유일 상장법인이라는 점에서 국내 투자자들에게도 많은 주목을 받을 것으로 전망된다.넥슨지티와 넷게임즈의 합병은 오는 2022년 2월 8일 주주총회를 거쳐 최종 결정된다. 합병 기일은 같은 해 3월 31일이다. 합병비율은 1 대 1.0423647(넷게임즈:넥슨지티)로 합병에 따른 존속회사는 넷게임즈이며, 신규 법인명은 넥슨게임즈(가칭)다.두 회사는 이번 합병을 통해 급변하는 글로벌 게임 시장에서 각각의 개발 법인이 가진 성공 노하우와 리소스를 결합해 PC, 모바일, 콘솔 등 멀티플랫폼을 지향하는 최상의 개발 환경을 구축할 계획이다. 넥슨게임즈의 대표이사는 현 넷게임즈 박용현 대표가 선임될 예정이며, 넥슨지티 신지환 대표는 등기이사직을 맡는다. 넥슨게임즈 이사진에는 넥슨코리아 이정헌 대표도 합류해 넥슨코리아와 협업도 강화할 계획이다.넷게임즈는 모바일 RPG ‘히트’와 ‘V4’를 통해 두 번의 대한민국 게임대상 수상 및 ‘오버히트’와 ‘블루아카이브’ 등을 통해 국내·외 모바일게임 시장에 굵직한 족적을 남긴 RPG 전문 개발사다. 넥슨지티는 FPS 게임 ‘서든어택’ 개발사로 슈팅 게임 명가로 자리매김했다. 올해로 서비스 16주년을 맞이했음에도 탁월한 라이브 운영으로 지난 3분기에만 전년 동기 대비 211%의 매출 성장을 기록했다.넥슨은 이번 합병으로 넥슨코리아 신규개발본부, 네오플, 넥슨게임즈, 원더홀딩스와 설립한 합작법인(니트로 스튜디오, 데브캣) 등을 큰 축으로 신규 개발을 이끌어갈 계획이다."
        split_news_text = bm25_ner_serving.preprocess_input_news(news_text, length=500)# 500글자 단위로 잘라 2차원 배열을 만들고, 공백 기준으로 split
        end = time()
        print("뉴스 데이터 전처리 시간: ", end - start)
        print()

        print("-"*20 + "query" + "-"*20)
        print(news_text)
        print("-"*40 + "\n")


        NUM_PREDICTION = 20
        REPORT_LEN = 200


        start = time()
        # 5.1.ner+bm25 모델 활용 관련주 추출
        ner_keywords = bm25_ner_serving.predict_ner(split_news_text, pred_config, args, tokenizer, pad_token_label_id, device, model, label_lst)
        ner_result, preprocessed_ner_keywords = bm25_ner_serving.print_corp_name(news_text, dart_dict, bm25, pre_dart, pre_name, mecab_tokenizer, ner=ner_keywords, num_prediction = NUM_PREDICTION)
        end = time()
        print("-"*20 + "ner bm25" + "-"*20)
        print("ner bm25: ", ner_result)
        print("ner bm25 inference 시간: ", end - start)
        print("ner keywords: ", preprocessed_ner_keywords)
        print("-"*40 + "\n")


        # 5.2.kwordrank+bm25 모델 활용 관련주 추출
        start = time()
        kr_result, kr_keywords = bm25_ner_serving.print_corp_name(news_text, dart_dict, bm25, pre_dart, pre_name, mecab_tokenizer, ner=None, num_prediction = NUM_PREDICTION)
        end = time()
        print("-"*20 + "kr_bm25" + "-"*20)
        print("kr_bm25: ", kr_result)
        print("kr keywords: ", kr_keywords)

        print("kr bm25 inference 시간:", end - start)
        print("-"*40 + "\n")

        # doc2vec
        start = time()
        doc2vec_result = doc2vec_model.doc2vec_inference(orig_news_text, num_prediction = NUM_PREDICTION)
        end = time()
        print("-"*20 + "doc2vec" + "-"*20)
        print("doc2vec :", doc2vec_result)
        print("doc2vec inference 시간:", end - start)
        print("-"*40 + "\n")

        from collections import defaultdict
        from functools import cmp_to_key
        count_dict = defaultdict(int)
        min_index_dict = {}


        if kr_result:
            zip_enum = zip(ner_result, kr_result, doc2vec_result)
        else:
            zip_enum = zip(ner_result, doc2vec_result)

        for i, methods_list in enumerate( zip_enum ):

            for corp_of_method in list(methods_list):
                
                count_dict[corp_of_method] += 1

                if corp_of_method not in min_index_dict:
                    min_index_dict[corp_of_method] = float("inf")

                min_index_dict[corp_of_method] = min(min_index_dict[corp_of_method], i)
        
        def comp(x, y):
            # if count dict value is same
            if x[1] == y[1]: 
                # smaller index affects more wieght
                if min_index_dict[x[0]] > min_index_dict[y[0]]:
                    return 1
                else:
                    return -1
            elif x[1] > y[1]:
                return 1
            else:
                return -1


        if kr_keywords == None:
            kr_keywords = []

        keywords_set = list(set(preprocessed_ner_keywords + kr_keywords))
        for key in keywords_set:
            if key in count_dict:
                count_dict[key] += 1
        
        infer_dict[idx]['keywords_set'] = keywords_set

        sorted_count_dict = dict(sorted(count_dict.items(), key = cmp_to_key(comp)))
        print(sorted_count_dict)
        infer_dict[idx]['sorted_count_dict'] = sorted_count_dict
    


    with open('infer_dict.json', 'w', encoding='UTF-8-sig') as file:
        file.write(json.dumps(infer_dict, ensure_ascii=False))
    # print(json_val)

# {
#     idx : {
#         'news_text' : str,
#         'ner_keywords' : list[str],
#         'corp_recommendation' : list[(str, int)]
#     }
# }