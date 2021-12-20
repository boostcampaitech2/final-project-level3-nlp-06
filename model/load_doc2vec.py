from gensim.models import doc2vec
import pandas as pd
from konlpy.tag import Mecab
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
from datasets import load_dataset

def load_doc_model():
    df =  load_dataset("nlpHakdang/beneficiary", data_files = "dart_preprocessed_original.csv", use_auth_token= "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")
    df = pd.DataFrame(df['train'])
    mecab = Mecab()

    tagged_corpus_list = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row['clean']
        tag = row['기업 이름']
        tagged_corpus_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text)))
    model = doc2vec.Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, workers=8, window=8)

    # Vocabulary 빌드
    model.build_vocab(tagged_corpus_list)
    # print(f"Tag Size: {len(model.docvecs.doctags.keys())}", end=' / ')

    # Doc2Vec 학습
    print("training model...")
    model.train(tagged_corpus_list, total_examples=model.corpus_count, epochs=50)

    # 모델 저장
    print("saving model...")
    model.save('./dart_fin.doc2vec')

    return doc2vec.Doc2Vec.load('dart_fin.doc2vec')

if __name__ == '__main__':
    load_doc_model()