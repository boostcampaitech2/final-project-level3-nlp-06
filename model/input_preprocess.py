import pandas as pd
import kss
import re
# from pykospacing import Spacing
# from hanspell import spell_checker
# 스펠링 보정
import time
from konlpy.tag import Mecab 

# 이거 뉴스 들어왔을 때 전처리되게 실행하려면?

def remove_email(texts):
    """
    이메일과 출처 사이트를 제거합니다.
    ``홍길동 abc@gmail.com 연락주세요!`` -> ``홍길동  연락주세요!``
    "출처 : http://www.thelec.kr"
    """
    preprocessed_text = []
    for text in texts:
        text = re.sub(r"[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", text).strip()
        text = re.sub(r"http://www+[a-zA-Z0-9-.]+\.[a-zA-Z0-9-.]+", "", text)
        if "출처 :" not in text:
            preprocessed_text.append(text)

    return preprocessed_text

def morph_filter(texts):
    """
    명사(NN), 동사(V), 형용사(J)의 포함 여부에 따라 문장 필터링
    """
    NN_TAGS = ["NNG", "NNP", "NNB", "NP"]
    V_TAGS = ["VV", "VA", "VX", "VCP", "VCN", "XSN", "XSA", "XSV"]
    J_TAGS = ["JKS", "J", "JO", "JK", "JKC", "JKG", "JKB", "JKV", "JKQ", "JX", "JC", "JKI", "JKO", "JKM", "ETM"]
    mecab = Mecab() 
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
