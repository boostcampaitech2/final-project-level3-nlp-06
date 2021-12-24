#!/usr/bin/env python
# coding: utf-8
import re
from datasets import load_dataset

from konlpy.tag import Mecab
from rank_bm25 import BM25Okapi
from krwordrank.word import summarize_with_keywords

from .utils import load_tokenizer, get_labels
from .predict import *

# 입력되는 뉴스데이터 전처리 함수
def split_news_for_bert(news_text, length):
    return [news_text[i:i+length].split() for i in range(0, len(news_text), length)]

class CustomArg:
    def __init__(self):
        pass
        self.model_dir = None#model_dir
        self.data_dir =  None#data_dir
        self.label_file =  None#label_file
        self.batch_size =  None#batch_size
        self.no_cuda =  None#no_cuda
    def add_argument(self, att_name, att_val = None, default = None, type = type, help = help):
        setattr(self, att_name, att_val)
        
# NER 모델 로드 함수
def load_NER():
    # ## 2. NER
    parser = CustomArg()
    # parser.add_argument("model_dir", att_val="/opt/ml/final-project-level3-nlp-06/KoBERT/model", type=str, help="Path to save, load model")
    parser.add_argument("model_dir", att_val="/opt/ml/coding/KoBERT/model", type=str, help="Path to save, load model")
    # parser.add_argument("data_dir", att_val="/opt/ml/final-project-level3-nlp-06/KoBERT/data", type=str, help="Ner model data files dir")
    parser.add_argument("data_dir", att_val="/opt/ml/coding/KoBERT/data", type=str, help="Ner model data files dir")
    parser.add_argument("label_file", att_val="label.txt", type=str, help="Ner model data files dir")

    parser.add_argument("batch_size", att_val=1, type=int, help="Batch size for prediction")
    parser.add_argument("no_cuda", att_val="store_false", default=False, help="Avoid using CUDA when available")
    pred_config = parser

    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    label_lst = get_labels(pred_config)



    logger.info(args)

    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = load_tokenizer(args)
    return pred_config, args, tokenizer, pad_token_label_id, device, model, label_lst

# NER 모델 평가 함수
def predict_ner(split_news_text, pred_config, args, tokenizer, pad_token_label_id, device, model, label_lst):
    lines = split_news_text
    dataset = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)
    all_slot_label_mask = None
    preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(preds.shape[0])]

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                preds_list[i].append(slot_label_map[preds[i][j]])
    
    # return to pred word list
    line = []
    for words, preds in zip(lines, preds_list):
        for word, pred in zip(words, preds):
            if pred == 'O' or pred =='UNK':
                pass
            else:
                if not "기자" in word:
                    line.append(word)

    return "{}\n".format(line)

# dart 데이터 전처리 함수
def preprocess_dart(dart_dataset):
    contents_list = ['회사의 개요',  '사업의 개요']
    dart_dict= { "corp_name" :[], "corp_text":[]}

    for corp in dart_dataset['train']:
        dart_text = ""
        dart_dict['corp_name'].append(corp['기업 이름'])
        for content in contents_list:
            dart_text += corp[content]
        dart_dict['corp_text'].append(dart_text)


    def preprocessing(s): 
            hangul = re.compile('[^ ㄱ-ㅣ가-힣+]')
            result = hangul.sub('', s).strip()
            return result

    pre_dart = {}
    for idx in range(len(dart_dict['corp_name'])):
        pre_dart[dart_dict['corp_name'][idx]] = preprocessing(dart_dict['corp_text'][idx])
        
    pre_name = {}
    for idx in range(len(dart_dict['corp_name'])):
        pre_name[preprocessing(dart_dict['corp_text'][idx])] = dart_dict['corp_name'][idx]

    return dart_dict, pre_dart, pre_name

# 키워드 + BM25 활용 관련주 출력 함수
def print_corp_name(news_text, dart_dict, bm25, pre_dart, pre_name, mecab_tokenizer, ner=None, num_prediction = 5):
    """
        tokenized_query : keywords
        inside_corp : 언급된 회사
        " ".join(target) : 뉴스텍스트
        corps : topk의 회사이름
        list_dart_n : topk dart
    """

    target = news_text
    inside_corp = []
    stopwords=['모두','지난','있다','회의','강조','성과','오는','모집','과정','계약',"체결","건립",'현대식','사업',
            '주관사로','통해','일까지','이벤트','지원','지역','기자','실험','한다','진행','설명회','예정이다',
            '이번','방문','올해','제공','찾아가','재림','당시','한국','프로젝트','있는','추진','다양한','적극',
            '위해','나갈','협력','협약','상당의','물품','명으로','지난해','증가했다','의한','영향','보면','통계',
            '순위','명당','관련','업체','부실','관심','명칭','합작법인','일자리','시흥배곧','서울대','했다',
            '한다고','위원장은','기반','혁신','인재','마련','정책','권고안','함께','시장','성장','대비','전월',
            '전년','동월','물가','하락','소비자','상승','운영','증가','억원','억만원','할인','사용','것으']

    if ner==None:
        try:
            keywords = summarize_with_keywords([target], min_count=4, max_length=7, beta=0.85, max_iter=10,stopwords=stopwords,verbose=True)
            tokenized_query = list(keywords.keys())
        except:
            tokenized_query = list()
    else:
        tokenized_query = mecab_tokenizer.nouns(ner)

    for key in target.split(" "):
        if key in dart_dict['corp_name']:
            inside_corp.append(key)
    

    for key in tokenized_query:
        if key in dart_dict['corp_name']:
            inside_corp.append(key)
    
    inside_corp = list(set(inside_corp))   
    doc_scores = bm25.get_scores(tokenized_query)
    
    if max(doc_scores) <3: #마땅한 회사가 없음
        return None, None, []
    else:
        list_dart_n = bm25.get_top_n(tokenized_query, list(pre_dart.values()), n=num_prediction)
        corps = [pre_name[dart] for dart in list_dart_n]
    return corps,tokenized_query, inside_corp


def get_preprocessed_dart_data():
    # 1. 다트데이터셋 전처리
    dart_dataset = load_dataset('nlpHakdang/beneficiary',  data_files="dart_v3_3.csv", use_auth_token='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')
    dart_dict, pre_dart, pre_name = preprocess_dart(dart_dataset)

    return dart_dict, pre_dart, pre_name


def load_bm25_model(preprocess_dart_data):
    # 2. bm25 모델 로드
    mecab_tokenizer = Mecab() # 전처리한 데이터에서 명사만 추출
    tot = [mecab_tokenizer.nouns(txt) for txt in tqdm(list(preprocess_dart_data.values()))]
    bm25 = BM25Okapi(tot)

    return mecab_tokenizer, bm25




import re
from time import time
class Bm25_module():
    def __init__(self, num_prediction = 5):
        # disable huggingface tokenizer multiprocess warning
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # # 1. 다트데이터셋 전처리
        self.dart_dict, self.corp_to_text_dict, self.text_to_corp_dict = get_preprocessed_dart_data()

        # # 2. bm25 모델 로드
        self.mecab_tokenizer, self.bm25 = load_bm25_model(self.corp_to_text_dict)

        # # 3. NER 모델 로드
        pred_config, args, tokenizer, pad_token_label_id, device, model, label_lst = load_NER()
        self.ner_config = {
                            "pred_config" : pred_config, "args" :args , "tokenizer" :tokenizer , 
                            "pad_token_label_id" :pad_token_label_id , "device" :device , 
                            "model" :model , "label_lst" :label_lst 
                        }


        self.cur_news_text = ""
        self.NUM_PREDICTION = num_prediction

    def preprocess_news(self, content):
        re_pattern = r"[가-힣]{0,4} (기자|선임기자|수습기자|특파원|객원기자|논설고문|통신원|연구소장)" # 이름 + 기자

        content = re.sub("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-]+)", "", content)
        content = re.sub("[-=+,#/\?:^@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·▲△▶■]", " ", content)
        content = re.sub(re_pattern, "", content).strip()
        content = re.sub("(.kr)","", content)
        content = ' '.join(content.split())

        return content

    def check_update_preprocess(self, news_text):
        if news_text != self.cur_news_text:
            self.cur_news_text = news_text

            self.preprocessed_news_text = self.preprocess_news(news_text)
        return self.preprocessed_news_text
    
    def inference_with_word_rank(self, news_text):
        preprocessed_news_text = self.check_update_preprocess(news_text)
        kr_result, kr_keywords, inside_corp = self.search_sim_corp_with_bm25(preprocessed_news_text)
        return kr_result, kr_keywords, inside_corp

    def ner_extract_keywords(self, split_news_text):
        
        return predict_ner(split_news_text, self.ner_config['pred_config'], 
                                            self.ner_config['args'], self.ner_config['tokenizer'], self.ner_config['pad_token_label_id'], 
                                            self.ner_config['device'], self.ner_config['model'], self.ner_config['label_lst']
                                            )

    def search_sim_corp_with_bm25(self, preprocessed_news_text, ner_keywords = None):
        related_corps, preprocessed_ner_keywords, inside_corp = print_corp_name(preprocessed_news_text, self.dart_dict, self.bm25, self.corp_to_text_dict, self.text_to_corp_dict, self.mecab_tokenizer, ner=ner_keywords, num_prediction = self.NUM_PREDICTION)
        return related_corps, preprocessed_ner_keywords, inside_corp

    def inference_with_ner(self, news_text):
        
        preprocessed_news_text = self.check_update_preprocess(news_text)

        split_news_text = split_news_for_bert(preprocessed_news_text, length=500)
        ner_keywords = self.ner_extract_keywords(split_news_text)
        related_corps, preprocessed_ner_keywords, inside_corp = self.search_sim_corp_with_bm25(preprocessed_news_text, ner_keywords = ner_keywords)
        return related_corps, preprocessed_ner_keywords, inside_corp


if __name__=="__main__":

    # # 1. 다트데이터셋 전처리
    dart_dict, pre_dart, pre_name = get_preprocessed_dart_data()

    # # 2. bm25 모델 로드
    mecab_tokenizer, bm25 = load_bm25_model(pre_dart)

    # # 3. NER 모델 로드
    pred_config, args, tokenizer, pad_token_label_id, device, model, label_lst = load_NER()


    from time import time

    start = time()
    ## 매번 실행헤서 결과를 얻는 부분
    # 4. 입력받은 뉴스데이터
    news_text = "‘넥슨’ 이름을 단 시가총액 1조원이 넘는 개발사가 조만간 모습을 드러낼 전망이다. 넥슨은 최근 개발 자회사 넷게임즈와 넥슨지티를 합병한다고 밝혔다. 모바일게임과 PC 온라인게임에 각각 강점을 가지고 있는 두 회사의 합병을 통해 시너지 효과를 낼 것으로 보인다. 아울러 넥슨의 국내 유일 상장법인이라는 점에서 국내 투자자들에게도 많은 주목을 받을 것으로 전망된다.넥슨지티와 넷게임즈의 합병은 오는 2022년 2월 8일 주주총회를 거쳐 최종 결정된다. 합병 기일은 같은 해 3월 31일이다. 합병비율은 1 대 1.0423647(넷게임즈:넥슨지티)로 합병에 따른 존속회사는 넷게임즈이며, 신규 법인명은 넥슨게임즈(가칭)다.두 회사는 이번 합병을 통해 급변하는 글로벌 게임 시장에서 각각의 개발 법인이 가진 성공 노하우와 리소스를 결합해 PC, 모바일, 콘솔 등 멀티플랫폼을 지향하는 최상의 개발 환경을 구축할 계획이다. 넥슨게임즈의 대표이사는 현 넷게임즈 박용현 대표가 선임될 예정이며, 넥슨지티 신지환 대표는 등기이사직을 맡는다. 넥슨게임즈 이사진에는 넥슨코리아 이정헌 대표도 합류해 넥슨코리아와 협업도 강화할 계획이다.넷게임즈는 모바일 RPG ‘히트’와 ‘V4’를 통해 두 번의 대한민국 게임대상 수상 및 ‘오버히트’와 ‘블루아카이브’ 등을 통해 국내·외 모바일게임 시장에 굵직한 족적을 남긴 RPG 전문 개발사다. 넥슨지티는 FPS 게임 ‘서든어택’ 개발사로 슈팅 게임 명가로 자리매김했다. 올해로 서비스 16주년을 맞이했음에도 탁월한 라이브 운영으로 지난 3분기에만 전년 동기 대비 211%의 매출 성장을 기록했다.넥슨은 이번 합병으로 넥슨코리아 신규개발본부, 네오플, 넥슨게임즈, 원더홀딩스와 설립한 합작법인(니트로 스튜디오, 데브캣) 등을 큰 축으로 신규 개발을 이끌어갈 계획이다."
    split_news_text = split_news_for_bert(news_text, length=500)# 500글자 단위로 잘라 2차원 배열을 만들고, 공백 기준으로 split
    preprocessed_time = time()
    print("뉴스 데이터 전처리 시간: ", preprocessed_time - start)

    # 5.1.ner+bm25 모델 활용 관련주 추출
    ner_result = predict_ner(split_news_text, pred_config, args, tokenizer, pad_token_label_id, device, model, label_lst)
    print_corp_name(news_text, dart_dict, bm25, pre_dart, pre_name, mecab_tokenizer, ner = ner_result)
    ner_bm25_time = time()
    print("ner bm25 inference 시간: ", ner_bm25_time - preprocessed_time)
    
    # 5.2.kwordrank+bm25 모델 활용 관련주 추출
    print_corp_name(news_text, dart_dict, bm25, pre_dart, pre_name, mecab_tokenizer, ner = None)
    kr_bm25_time = time()
    print("kr bm25 inference 시간:", kr_bm25_time - ner_bm25_time)