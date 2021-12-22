#!/usr/bin/env python3
from model import doc2vec_model
from KoBERT.bm25_ner_serving import Bm25_module


class Models():
    def __init__(self, num_prediction):
        self.num_prediction = num_prediction
        self.bm25_module = Bm25_module(num_prediction = num_prediction)

    def bm25_inference_with_ner(self, news_text):
        return self.bm25_module.inference_with_ner(news_text)
    def bm25_inference_with_word_rank(self, news_text):
        return self.bm25_module.inference_with_word_rank(news_text)
    def doc2vec_inference(self, news_text):
        return doc2vec_model.doc2vec_inference(news_text, num_prediction = self.num_prediction)

def run_ensemble(wordrank_sim_corps, ner_sim_corps, doc2vec_sim_corps, ner_keywords, wordrank_keywords):
        
        from collections import defaultdict
        from functools import cmp_to_key
        count_corp_dict = defaultdict(int)
        min_index_dict = {}

        """
        앙상블 규칙: 아이디어 from jacksson song
        - ner, wordrank, doc2vec의 세가지 방법에서 유사 기업 이름이 나옴. ner과 wordrank에서 키워드가 나옴.
        - 기업 이름 앙상블: 
            -- 전제: 앞에 나올 수록 유사성이 높은 기업. 따라서 index가 중요
            -- 3가지 각 방법에서 동일한 기업이 나오면 높은 유사성을 가진 기업
            => 각 방법에서 나온 기업의 빈도 수 카운트
            => 키워드들을 모아서 set으로 유일하게 만들고, 그 안에서 기업 이름이 포함되어 있으면 카운트 + 1

            
            => 카운트가 높은 순서대로 유사성이 높은 기업으로 설정
            => 카운트가 동일하면, 각 기업들이 각 방법에서 최소 index 순서로 정렬
        
        """


        if wordrank_sim_corps: # wordrank의 경우 뉴스 입력 데이터에 따라 핵심 키워드가 없음
            zip_enum = zip(ner_sim_corps, wordrank_sim_corps, doc2vec_sim_corps)
        else:
            zip_enum = zip(ner_sim_corps, doc2vec_sim_corps)

        for i, methods_list in enumerate( zip_enum ):

            for corp_of_method in list(methods_list):
                
                count_corp_dict[corp_of_method] += 1

                if corp_of_method not in min_index_dict: # 처음 보는 기업이면 index 무한대로 초기화. 
                    min_index_dict[corp_of_method] = float("inf")

                min_index_dict[corp_of_method] = min(min_index_dict[corp_of_method], i) # 각 기업의 각 method에서 최소 index 업데이트
        
        def comp(x, y):
            # if count dict value is same
            if x[1] == y[1]: 
                # smaller index affects more wieght
                # 두 기업의 count가 같으면, 각 방법에서의 최소 인덱스로 유사도 비교. index가 낮을 수록 유사도가 높다
                if min_index_dict[x[0]] > min_index_dict[y[0]]:
                    return 1
                else:
                    return -1
            elif x[1] > y[1]:
                return 1
            else:
                return -1


        if wordrank_keywords == None:
            wordrank_keywords = []

        keywords_set = list(set(ner_keywords + wordrank_keywords))
        for key in keywords_set:
            if key in count_corp_dict:
                count_corp_dict[key] += 1
        

        ensembled_sim_corps_to_score_dict = dict(sorted(count_corp_dict.items(), key = cmp_to_key(comp)))
        return ensembled_sim_corps_to_score_dict, keywords_set


def ensemble_inference_realtime(models, news_text):
    n_c, n_k = models.bm25_inference_with_ner(news_text)
    k_c, k_k = models.bm25_inference_with_word_rank(news_text)
    d_c = models.doc2vec_inference(news_text)

    ensembled_sim_corps_to_score_dict, keywords_set = run_ensemble(k_c, n_c, d_c, n_k, k_k)

    return ensembled_sim_corps_to_score_dict, keywords_set

def ensemble_inference_from_disk(models, query_dir = 'queries.txt', ):
    infer_dict = {}
    file1 = open(query_dir, 'r')
    Lines = file1.readlines()
    idx = -1
    
    for line in Lines:
        if line == "\n":
            continue
        news_text = line.strip()
        idx += 1
        infer_dict[idx] = {}
        infer_dict[idx]['news_text'] = news_text

        ensembled_sim_corps_to_score_dict, keywords_set = ensemble_inference_realtime(models, news_text)

        infer_dict[idx]['keywords_set'] = keywords_set
        infer_dict[idx]['sorted_count_dict'] = ensembled_sim_corps_to_score_dict
    

    import json
    with open('infer_dict.json', 'w', encoding='UTF-8-sig') as file:
        file.write(json.dumps(infer_dict, ensure_ascii=False))


def ensemble_inference_console_test(models):
    while True:
        news_text = input("input news text: 뉴스 기사 개생 없이 넣으세요\n")
        if news_text == "\n":
            break
        ensembled_sim_corps_to_score_dict, keywords_set = ensemble_inference_realtime(models, news_text)
        print(ensembled_sim_corps_to_score_dict.keys())


def inference_each_model_and_save_results(models):
    query_dir = "queries.txt"



    infer_word_rank_dict = {}
    infer_ner_dict = {}
    infer_doc2vec_dict = {}


    file1 = open(query_dir, 'r')
    Lines = file1.readlines()
    idx = -1
    
    for line in Lines:
        if line == "\n":
            continue
        news_text = line.strip()
        idx += 1
        infer_word_rank_dict[idx] = {}
        infer_word_rank_dict[idx]['news_text'] = news_text

        infer_ner_dict[idx] = {}
        infer_ner_dict[idx]['news_text'] = news_text

        infer_doc2vec_dict[idx] = {}
        infer_doc2vec_dict[idx]['news_text'] = news_text


        k_c, k_k = models.bm25_inference_with_word_rank(news_text)
        n_c, n_k = models.bm25_inference_with_ner(news_text)
        d_c = models.doc2vec_inference(news_text)



        sorted_count_dict = {}
        if k_c:
            for i, corp_name in enumerate( k_c ):
                sorted_count_dict[corp_name] = len(k_c) - i - 1
        sorted_count_dict = dict(sorted(sorted_count_dict.items(), key = lambda x:x[1])) # corp with score. the larger, the better.

        infer_word_rank_dict[idx]['keywords_set'] = k_k
        infer_word_rank_dict[idx]['sorted_count_dict'] = sorted_count_dict

        sorted_count_dict = {}
        for i, corp_name in enumerate( n_c ):
            sorted_count_dict[corp_name] = len(n_c) - i - 1

        infer_ner_dict[idx]['keywords_set'] = n_k
        infer_ner_dict[idx]['sorted_count_dict'] = dict(sorted(sorted_count_dict.items(), key = lambda x:x[1]))

        sorted_count_dict = {}
        for i, corp_name in enumerate( n_c ):
            sorted_count_dict[corp_name] = len(n_c) - i - 1
        infer_doc2vec_dict[idx]['sorted_count_dict'] = dict(sorted(sorted_count_dict.items(), key = lambda x:x[1]))
    

    import json
    with open('infer_word_rank_dict.json', 'w', encoding='UTF-8-sig') as file:
        file.write(json.dumps(infer_word_rank_dict, ensure_ascii=False))

    with open('infer_ner_dict.json', 'w', encoding='UTF-8-sig') as file:
        file.write(json.dumps(infer_ner_dict, ensure_ascii=False))

    with open('infer_doc2vec_dict.json', 'w', encoding='UTF-8-sig') as file:
        file.write(json.dumps(infer_doc2vec_dict, ensure_ascii=False))



if __name__ == '__main__':
    news_text = "[헤럴드경제=이호 기자] '3번째 인터넷은행'인 토스뱅크(가칭)가 이번주 본인가를 신청한다는 소식에 토스뱅크의 10% 지분을 보유한 컨소시엄에 모회사인 이랜드월드가 참여해 이월드가 주목을 받고 있다. 28일 한국거래소에 따르면 이월드는 오후 1시 30분 전일 대비 35원(0.88%) 하락한 3370원에 거래되고 있다.오는 7월 출범을 앞둔 토스뱅크는 금융감독원에 인터넷전문은행 본인가를 신청하고 서비스 출시에 박차를 가하고 있는 상황이다. 비바리퍼블리카(토스)는 이번주 중 인터넷전문은행 본인가를 신청할 것으로 알려졌다. 2019년 12월 토스뱅크가 예비인가를 받은 지 1년 1개월만으로 토스뱅크는 금융위원회로부터 오는 3월 본인가를 받겠다는 목표를 세웠다.토스뱅크의 최대주주는 지분 34%를 보유한 토스이고, 지분 10%를 보유한 하나은행·한화투자증권·중소기업중앙회·이랜드월드는 2대 주주로 참여한다. 기타 주주로는 SC제일은행과 웰컴저축은행 등이 있다.한편, 지난 2019년 10월 이랜드는 패션 계열사 이랜드월드는 비바리퍼블리카(토스) 중심 제3인터넷전문은행 컨소시엄에 참여했다. 컨소시엄 참여로 그룹의 핵심 사업인 패션·유통에 금융을 접목하겠다 게 이랜드의 복안이다. 이랜드는 토스와 협업을 통해 그룹의 멤버십 회원들에게 생활 금융 서비스를 제공할 수 있을 것이라고 밝힌 바 있다. 신세계푸드는 가정간편식 ‘올반’과 맥주 브랜드 ‘칭따오’가 협업한 '칭따오엔 왕교자' 2종을 출시한다고 20일 밝혔다. 신세계푸드는 코로나19 장기화로 혼술, 홈술 트렌드가 강화되는 것에 주목했다. 특히 궁합이 맞는 음식을 의미하는 ‘푸드 페어링’에 대한 소비자들의 관심이 높아지면서 유명 주류 브랜드와의 협업을 통해 간편식을 선보인다면 홈술족의 호응을 이끌어 낼 것으로 예상했다. 이에 신세계푸드는 맥주 브랜드 ‘칭따오’와 손잡고 맥주 안주로 제격인 만두를 활용해 개발한 ‘칭따오엔 양꼬치맛 왕교자’, ‘칭따오엔 마라 왕교자’ 등 2종을 선보였다. ‘칭따오엔 양꼬치맛 왕교자’는 국내산 돼지고기와 양고기의 육즙, 쯔란, 코리앤더(고수) 파우더 등 이국적인 향신료의 조합으로 속을 채워 양꼬치의 풍미와 식감을 맛볼 수 있는 것이 특징이다. ‘칭따오엔 마라 왕교자’는 국내산 돼지고기에 채 썬 오징어를 더해 쫄깃한 맛을 내며, 마라 향신료를 더해 톡 쏘는 알싸함을 느낄 수 있다. 신세계푸드는 ‘올반’과 ‘칭따오’의 협업을 기념해 오는 22일 오후 8시 네이버 쇼핑라이브를 진행하고 ‘칭따오엔 왕교자’를 넉넉하게 즐길 수 있도록 만든 ‘칭따오엔 양꼬치맛 군만두’, ‘칭따오엔 마라 군만두’ 패키지를 할인 판매한다. 아울러 ‘칭따오엔 왕교자’ 2종은 전국 CU에서 만나볼 수 있다. 신세계푸드 관계자는 “푸드 페어링을 추구하는 홈술 트렌드에 맞춰 맥주와 어울리는 올반 가정간편식을 선보이기 위해 칭따오와 함께 협업을 진행하게 됐다”며 “점점 세분화되어가는 소비자들의 입맛을 사로잡기 위해 식문화 트렌드를 반영한 가정간편식 라인업 확장 뿐 아니라 다양한 협업 마케팅을 강화해 나가겠다”고 말했다.[헤럴드경제=이호 기자] '3번째 인터넷은행'인 토스뱅크(가칭)가 이번주 본인가를 신청한다는 소식에 토스뱅크의 10% 지분을 보유한 컨소시엄에 모회사인 이랜드월드가 참여해 이월드가 주목을 받고 있다. 28일 한국거래소에 따르면 이월드는 오후 1시 30분 전일 대비 35원(0.88%) 하락한 3370원에 거래되고 있다.오는 7월 출범을 앞둔 토스뱅크는 금융감독원에 인터넷전문은행 본인가를 신청하고 서비스 출시에 박차를 가하고 있는 상황이다. 비바리퍼블리카(토스)는 이번주 중 인터넷전문은행 본인가를 신청할 것으로 알려졌다. 2019년 12월 토스뱅크가 예비인가를 받은 지 1년 1개월만으로 토스뱅크는 금융위원회로부터 오는 3월 본인가를 받겠다는 목표를 세웠다.토스뱅크의 최대주주는 지분 34%를 보유한 토스이고, 지분 10%를 보유한 하나은행·한화투자증권·중소기업중앙회·이랜드월드는 2대 주주로 참여한다. 기타 주주로는 SC제일은행과 웰컴저축은행 등이 있다.한편, 지난 2019년 10월 이랜드는 패션 계열사 이랜드월드는 비바리퍼블리카(토스) 중심 제3인터넷전문은행 컨소시엄에 참여했다. 컨소시엄 참여로 그룹의 핵심 사업인 패션·유통에 금융을 접목하겠다 게 이랜드의 복안이다. 이랜드는 토스와 협업을 통해 그룹의 멤버십 회원들에게 생활 금융 서비스를 제공할 수 있을 것이라고 밝힌 바 있다. 신세계푸드는 가정간편식 ‘올반’과 맥주 브랜드 ‘칭따오’가 협업한 '칭따오엔 왕교자' 2종을 출시한다고 20일 밝혔다. 신세계푸드는 코로나19 장기화로 혼술, 홈술 트렌드가 강화되는 것에 주목했다. 특히 궁합이 맞는 음식을 의미하는 ‘푸드 페어링’에 대한 소비자들의 관심이 높아지면서 유명 주류 브랜드와의 협업을 통해 간편식을 선보인다면 홈술족의 호응을 이끌어 낼 것으로 예상했다. 이에 신세계푸드는 맥주 브랜드 ‘칭따오’와 손잡고 맥주 안주로 제격인 만두를 활용해 개발한 ‘칭따오엔 양꼬치맛 왕교자’, ‘칭따오엔 마라 왕교자’ 등 2종을 선보였다. ‘칭따오엔 양꼬치맛 왕교자’는 국내산 돼지고기와 양고기의 육즙, 쯔란, 코리앤더(고수) 파우더 등 이국적인 향신료의 조합으로 속을 채워 양꼬치의 풍미와 식감을 맛볼 수 있는 것이 특징이다. ‘칭따오엔 마라 왕교자’는 국내산 돼지고기에 채 썬 오징어를 더해 쫄깃한 맛을 내며, 마라 향신료를 더해 톡 쏘는 알싸함을 느낄 수 있다. 신세계푸드는 ‘올반’과 ‘칭따오’의 협업을 기념해 오는 22일 오후 8시 네이버 쇼핑라이브를 진행하고 ‘칭따오엔 왕교자’를 넉넉하게 즐길 수 있도록 만든 ‘칭따오엔 양꼬치맛 군만두’, ‘칭따오엔 마라 군만두’ 패키지를 할인 판매한다. 아울러 ‘칭따오엔 왕교자’ 2종은 전국 CU에서 만나볼 수 있다. 신세계푸드 관계자는 “푸드 페어링을 추구하는 홈술 트렌드에 맞춰 맥주와 어울리는 올반 가정간편식을 선보이기 위해 칭따오와 함께 협업을 진행하게 됐다”며 “점점 세분화되어가는 소비자들의 입맛을 사로잡기 위해 식문화 트렌드를 반영한 가정간편식 라인업 확장 뿐 아니라 다양한 협업 마케팅을 강화해 나가겠다”고 말했다."

    models = Models(num_prediction = 20) # 필요한 모델과 데이터 로드. 서버 실행 시 1번만 수행
    # inference_each_model_and_save_results(models)
    
    ensembled_sim_corps_to_score_dict, keywords_set = ensemble_inference_realtime(models, news_text) # models object와 유저가 넣은 뉴스 기사 입력
    # print( ensembled_sim_corps_to_score_dict.keys() ) # 뒤로 갈수록 유사한 기업들

    # 주어진 기사들을 한꺼번에 inference 해서 결과를 파일로 저장
    # ensemble_inference_from_disk(models, query_dir = 'queries.txt') # query txt: 한 줄에 개행 없는 하나의 기사.

    # 콘솔에서 테스트: 
    #ensemble_inference_console_test(models)



