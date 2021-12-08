# final-project-level3-nlp-06

# 1. 프로젝트 개요

# 2. 디렉토리 구조
final-project-level3-nlp-06
.
├── convert_to_dataset - 뉴스,DART 데이터를 허깅페이스 데이터셋으로 변환
│   ├── aihub_news
│   │   └── aihub_news.py
│   ├── convert_news.ipynb
│   └── dataload_news.ipynb
├── data - 뉴스, DART 원본 데이터셋의 저장소
│   ├── dart
│   │   ├── corp_2500_result.pickle
│   │   ├── corporation_list.pickle
│   │   ├── dart_hug.ipynb
│   │   └── dataset_hug_preprocess_tutorial.ipynb
│   └── news
│       ├── MainData.pickle
│       ├── SubData.pickle
│       ├── train
│       │   └── train_original.json
│       └── valid
│           └── valid_original.json
├── data_collect - DART 데이터 수집 코드
│   └── get_dart.ipynb
├── data_preprocess - 데이터 전처리 코드 
├── EDA - 데이터 분석
│   ├── eda_dart.ipynb
│   └── eda_news30k.ipynb
├── model - 모델 구조 및 실험 저장
│   ├── bm25_test.ipynb
├── retriever_dart - 뉴스데이터를 통한 DART 문서 검색
└── text_classification - 뉴스 입력에 대한 주식관련 뉴스인지 분류