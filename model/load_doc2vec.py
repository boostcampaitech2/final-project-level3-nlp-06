from gensim.models import doc2vec
import pandas as pd
from konlpy.tag import Mecab
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
from datasets import load_dataset

def load_doc_model():

    df = pd.read_csv('dart_v2_1.csv') # 이건 만들어주시면 됩니다. row(2313개) corp_name, clean데이터만 있으면 됨
    mecab = Mecab()

    tagged_corpus_list = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row['clean']
        tag = row['corp_name']
        tagged_corpus_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text)))
    
    model = doc2vec.Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, workers=8, window=8)

    # Vocabulary 빌드
    model.build_vocab(tagged_corpus_list)
    print(f"Tag Size: {len(model.docvecs.doctags.keys())}", end=' / ')

    # Doc2Vec 학습
    model.train(tagged_corpus_list, total_examples=model.corpus_count, epochs=50)

    # 모델 저장
    model.save('dart.doc2vec')

    return doc2vec.Doc2Vec.load('dart.doc2vec')

