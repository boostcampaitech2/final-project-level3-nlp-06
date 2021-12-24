from gensim.models import doc2vec
import pandas as pd
from konlpy.tag import Mecab
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
import os
import kss
from .input_preprocess import remove_email,  morph_filter
from .load_doc2vec import *

# input이 doc
def preprocess_news(doc):
    # from koalanlp import API
    # from koalanlp.proc import SentenceSplitter

    # splitter = SentenceSplitter(splitter_type=API.HANNANUM)
    # paragraph = splitter("분리할 문장을 이렇게 넣으면 문장이 분리됩니다. 간단하죠?")
    # # 또는 splitter.sentences(...), splitter.invoke(...)

    # print(paragraph[0]) # == 분리할 문장을 이렇게 넣으면 문장이 분리됩니다.
    # print(paragraph[1]) # == 간단하죠?
    preprocessed_news = []
    

    sents = kss.split_sentences(doc, backend = "mecab", num_workers=-1)
    sents = remove_email(sents)
    sents = morph_filter(sents)
    sents = " ".join(sents)

    preprocessed_news.append(sents)
    
    return preprocessed_news

def tokenize(text):
    mecab = Mecab() 
    return mecab.nouns(str(text))


def doc2vec_inference(doc, num_prediction = 3):
    if os.path.join(os.getcwd(), 'KoBERT/model/dart_fin.doc2vec') :
        model = doc2vec.Doc2Vec.load(os.path.join(os.getcwd(), 'KoBERT/model/dart_fin.doc2vec')) # 경로를 바꿔주세요
    else:
        model = load_doc_model()

    text = preprocess_news(doc)
    docs_mod = tokenize(text)

    scriptV = model.infer_vector(docs_mod, alpha=0.025, min_alpha=0.025, epochs=50)

    result = model.docvecs.most_similar(positive=[scriptV], topn=num_prediction)
    total_result = [corp[0] for corp in result]
    return total_result

def main():
    # doc = input()
    # Example
    doc = "삼성전자가 고성능 솔리드스테이트드라이브(SSD)와 그래픽 D램 등 첨단 메모리 반도체를 글로벌 자동차 제조사에 공급한다고 16일 밝혔다."
    result = doc2vec_inference(doc)
    print(result)

if __name__ == '__main__':
    main()
