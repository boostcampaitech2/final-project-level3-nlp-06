import datasets
import json
from models import Corporations
# from database import db
from app import db


def insert_corporations():
    dartdataset = datasets.load_dataset('nlpHakdang/beneficiary',  data_files = "dart_ver1_2.csv", use_auth_token='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')
    for data in dartdataset['train']:
        db.session.add(Corporations(name=data['corp_name'], details=json.dumps(data)))
    db.session.commit()
    print("Insert Corporations Finish!")

