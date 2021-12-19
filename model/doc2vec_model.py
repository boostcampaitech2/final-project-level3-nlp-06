from gensim.models import doc2vec
import pandas as pd
from konlpy.tag import Mecab
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm

import kss
from input_preprocess import remove_email,  morph_filter


# input이 doc
def preprocess_news(doc):

    preprocessed_news = []
    sents = kss.split_sentences(doc)
    sents = remove_email(sents)
    sents = morph_filter(sents)
    sents = " ".join(sents)

    preprocessed_news.append(sents)
    
    return preprocessed_news

def tokenize(text):
    mecab = Mecab() 
    return mecab.nouns(str(text))

def main():
    doc = input()
    text = preprocess_news(doc)
    model = doc2vec.Doc2Vec.load('/opt/ml/final_project/data/dart/dart.doc2vec')
    docs_mod = tokenize(text)

    scriptV=model.infer_vector(docs_mod, alpha=0.025, min_alpha=0.025, epochs=50, steps=None)

    result = model.docvecs.most_similar(positive=[scriptV], topn=3)
    total_result = [corp[0] for corp in result]
    print(total_result)

if __name__ == '__main__':
    main()