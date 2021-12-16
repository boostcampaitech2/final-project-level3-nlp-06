import pandas as pd
import json
import kss
import re
from pykospacing import Spacing
from hanspell import spell_checker
# 스펠링 보정

from konlpy.tag import Mecab 



def spell_check_sent(texts):
    """
    맞춤법을 보정합니다.
    """
    preprocessed_text = []
    for text in texts:
        try:
            spelled_sent = spell_checker.check(text)
            checked_sent = spelled_sent.checked 
            if checked_sent:
                preprocessed_text.append(checked_sent)
        except:
            preprocessed_text.append(text)
    return preprocessed_text


def remove_email(texts):
    """
    이메일을 제거합니다.
    ``홍길동 abc@gmail.com 연락주세요!`` -> ``홍길동  연락주세요!``
    """
    preprocessed_text = []
    for text in texts:
        text = re.sub(r"[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", text).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text

def remove_press(texts):
    """
    언론 정보를 제거합니다.
    ``홍길동 기자 (연합뉴스)`` -> ````
    ``(이스탄불=연합뉴스) 하채림 특파원 -> ````
    """
    re_patterns = [
        r"\([^(]*?(뉴스|경제|일보|미디어|데일리|한겨례|타임즈|위키트리)\)",
        r"[가-힣]{0,4} (기자|선임기자|수습기자|특파원|객원기자|논설고문|통신원|연구소장) ",  # 이름 + 기자
        r"[가-힣]{1,}(뉴스|경제|일보|미디어|데일리|한겨례|타임|위키트리)",  # (... 연합뉴스) ..
        r"\(\s+\)",  # (  )
        r"\(=\s+\)",  # (=  )
        r"\(\s+=\)",  # (  =)
    ]

    preprocessed_text = []
    for text in texts:
        for re_pattern in re_patterns:
            text = re.sub(re_pattern, "", text).strip()
        if text:
            preprocessed_text.append(text)    
    return preprocessed_text

def spacing_sent(texts):
    """
    띄어쓰기를 보정합니다.
    """
    preprocessed_text = []
    for text in texts:
        text = spacing(text)
        if text:
            preprocessed_text.append(text)
    return preprocessed_text

def stop_word(file_path):

    with open(file_path) as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    return lines

def morph_filter(texts):
    """
    명사(NN), 동사(V), 형용사(J)의 포함 여부에 따라 문장 필터링
    """
    NN_TAGS = ["NNG", "NNP", "NNB", "NP"]
    V_TAGS = ["VV", "VA", "VX", "VCP", "VCN", "XSN", "XSA", "XSV"]
    J_TAGS = ["JKS", "J", "JO", "JK", "JKC", "JKG", "JKB", "JKV", "JKQ", "JX", "JC", "JKI", "JKO", "JKM", "ETM"]

    preprocessed_text = []
    for text in texts:
        morphs = mecab.pos(text, join=False)

        nn_flag = False
        v_flag = False
        j_flag = False
        for morph in morphs:
            pos_tags = morph[1].split("+")
            for pos_tag in pos_tags:
                if not nn_flag and pos_tag in NN_TAGS:
                    nn_flag = True
                if not v_flag and pos_tag in V_TAGS:
                    v_flag = True
                if not j_flag and pos_tag in J_TAGS:
                    j_flag = True
            if nn_flag and v_flag and j_flag:
                preprocessed_text.append(text)
                break
    return preprocessed_text

def remove_stopwords(sents, stopwords):
    #  큰 의미가 없는 불용어 정의
    preprocessed_text = []
    for sent in sents:
        sent = [w for w in sent.split(' ') if w not in stopwords]# 불용어 제거
        preprocessed_text.append(' '.join(sent))
    return preprocessed_text


def main():

    with open("train_original.json", "r",encoding='utf-8-sig') as st_json:
        train_original = json.load(st_json)

    with open("valid_original.json", "r",encoding='utf-8-sig') as st_json:
        valid_original = json.load(st_json)

    train_data = []
    for i in range(len(train_original['documents'])):
        if train_original['documents'][i]['category'] == '경제' or train_original['documents'][i]['category'] == '기업' :
            train_data.append(train_original['documents'][i])

    train_original = train_data

    valid_data = []
    for i in range(len(valid_original['documents'])):
        if valid_original['documents'][i]['category'] == '경제' or valid_original['documents'][i]['category'] == '기업' :
            valid_data.append(valid_original['documents'][i])

    valid_original = valid_data

    print("----------complete to make datasets------------")
    file_path = "stopword.txt"
    stopwords = stop_word(file_path)

    global spacing, mecab
    spacing = Spacing()
    mecab = Mecab() 
    train_texts = []
    cnt = 0
    for i in range(len(train_original)):
        data = train_original[i]['text']
        # i : 0 ~ 271093 까지 돌 것
        context = []
        # idx : data(각 document당의 idx 개수만큼 돌 것)

        for idx in range(0, len(data)):
            if data[idx] != []:
                context.append(data[idx][0]['sentence'])

        sents = []
        for sent in context:
            sent = sent.strip()
            if sent:
                splited_sent = kss.split_sentences(sent)
                sents.extend(splited_sent)
        sents = remove_email(sents)
        sents = remove_press(sents)
        sents = spacing_sent(sents)
        sents = spell_check_sent(sents)
        sents = morph_filter(sents)
        sents = remove_stopwords(sents, stopwords)
        sents = " ".join(sents)

        train_texts.append(sents)
        cnt += 1
        if cnt % 10 == 0 :
            print(f'{cnt}번 파일을 생성했습니다.')
    return train_texts


train_text = main()
df = pd.DataFrame(train_text)
df.to_csv('news_text.csv')

