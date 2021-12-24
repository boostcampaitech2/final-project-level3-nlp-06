import torch

import gluonnlp as nlp
import numpy as np
from torch import nn

from torch.utils.data import Dataset
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

    
class BERTClassifier(nn.Module):
    def __init__(self,
                    bert,
                    hidden_size = 768,
                    num_classes=2, ##주의: 클래스 수 바꾸어 주세요!##
                    dr_rate=None,
                    params=None):
        super(BERTClassifier, self).__init__()
        
        self.bert = bert
        self.dr_rate = dr_rate
                    
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

class BERTModels():
    def __init__(self):
        self.bertmodel, self.vocab = get_pytorch_kobert_model()
        # self.dr_rate = dr_rate
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.model = BERTClassifier(self.bertmodel, dr_rate=self.dr_rate).to(self.device)
        self.model = torch.load('./model.pt')
        self.tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), self.vocab)

    def predict(self, news):
        max_len = 64
        news_list = [[news, '0']]

        bert_dataset = BERTDataset(news_list, 0, 1, self.tokenizer, max_len, True, False)
        dataloader = torch.utils.data.DataLoader(bert_dataset, batch_size=1, num_workers=4)

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(dataloader): 
            token_ids = token_ids.long().to(self.device) 
            segment_ids = segment_ids.long().to(self.device) 
            valid_length = valid_length 
            out = self.model(token_ids, valid_length, segment_ids)
            prediction = out.cpu().detach().numpy().argmax()
            break
        return prediction