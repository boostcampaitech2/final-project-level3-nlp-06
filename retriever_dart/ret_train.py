#!/usr/bin/env python
# coding: utf-8

# In[24]:


import torch

import datasets
import numpy as np
import pandas as pd
# import tqdm.notebook as tq
from tqdm import tqdm as tq

import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename='ret_train.log', level=logging.DEBUG)


def log_print(msg):
    logging.info(msg)
    print(msg)
# In[25]:


# create data file from retrieal_dataset.ipynb


train_df = pd.read_csv("proto_train_comb_df.csv")
# train_df = train_df.iloc[:100]
train_df.head(), len(train_df)


# In[26]:


valid_df = pd.read_csv("proto_valid_comb_df.csv")
valid_df.head()


# In[27]:


# 기업 이름을 label으로 만든다. 나중에 table 넘겨서 pooling에 사용할 것
dic = {}
idx = 0
for c in train_df['기업 이름']:
    if c not in dic:
        dic[c] = idx
        idx += 1

val_dic = {}
idx = 0
for c in valid_df['기업 이름']:
    if c not in val_dic:
        val_dic[c] = idx
        idx += 1

valid_df['corp_code'] = valid_df['기업 이름'].map(lambda x:val_dic[x])
train_df['corp_code'] = train_df['기업 이름'].map(lambda x:dic[x])
train_df.head()


# In[28]:


from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    # AdamW, get_linear_schedule_with_warmup,
    # TrainingArguments,
)


# In[29]:


import transformers
print(transformers.__version__)


# In[30]:


tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")


# In[31]:


dart_tok = tokenizer( list(train_df['psg']) ,padding = "max_length", return_tensors = "pt", truncation = True) 
news_tok = tokenizer( list(train_df['news']) ,padding = "max_length", return_tensors = "pt", truncation = True) 

dart_valid_tok = tokenizer( list(valid_df['psg']) ,padding = "max_length", return_tensors = "pt", truncation = True) 
news_valid_tok = tokenizer( list(valid_df['news']) ,padding = "max_length", return_tensors = "pt", truncation = True) 


# In[32]:


corp_code = torch.tensor(train_df['corp_code'])
valid_corp_code = torch.tensor(valid_df['corp_code'])

# In[33]:


dart_tok['input_ids'].shape, news_tok['input_ids'].shape, corp_code.shape


# In[34]:


dart_tok.keys()


# In[35]:


from torch.utils.data import DataLoader, TensorDataset


# In[36]:


train_set = TensorDataset(
    news_tok['input_ids'], news_tok['token_type_ids'], news_tok['attention_mask'], \
    dart_tok['input_ids'], dart_tok['token_type_ids'], dart_tok['attention_mask'], \
    corp_code)

valid_set = TensorDataset(
    news_valid_tok['input_ids'], news_valid_tok['token_type_ids'], news_valid_tok['attention_mask'], \
    dart_valid_tok['input_ids'], dart_valid_tok['token_type_ids'], dart_valid_tok['attention_mask'], \
    valid_corp_code)


# In[37]:


BATCH_SIZE = 16
dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

val_dataloader = DataLoader(valid_set, batch_size=BATCH_SIZE,shuffle = True)

dataloader


# # 모델

# In[38]:


class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
      
    def forward(self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output


# In[39]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[40]:


news_encoder = BertEncoder.from_pretrained("klue/bert-base").to(device)
dart_encoder = BertEncoder.from_pretrained("klue/bert-base").to(device)



# # 학습

# In[41]:


from torch import optim
news_optimizer = optim.SGD(news_encoder.parameters(), lr=0.01, momentum=0.9)
dart_optimizer = optim.SGD(dart_encoder.parameters(), lr=0.01, momentum=0.9)

from torch.nn import CrossEntropyLoss


# In[42]:



epochs = 3
for e in range( epochs ):
    e_loss = 0
    num_correct = 0
    print(f"epoch {e+1} start...")
    print(f"{len(dataloader)} batches proceed...")
    for batch_idx, batch in tq( enumerate(dataloader) ):
        news_data = {
            'input_ids' : batch[0].to(device).to(torch.int64), # ( B, seq_len )
            'token_type_ids' : batch[1].to(device).to(torch.int64),
            'attention_mask' : batch[2].to(device).to(torch.int64),
        }
        
        dart_data = {
            'input_ids' : batch[3].to(device).to(torch.int64), # ( B, seq_len )
            'token_type_ids' : batch[4].to(device).to(torch.int64),
            'attention_mask' : batch[5].to(device).to(torch.int64),
        }

        corp_table = batch[6]


        corp_code = corp_table.tolist()
        dic = {}
        for i, corp in enumerate( corp_code ):
            if corp not in dic:
                dic[corp] = [i]
            else:
                dic[corp].append( i )


        batch_size = batch[0].shape[0]
        del batch
        
        news_encoder.train()
        news_encoder.zero_grad()
        news_emb = news_encoder(**news_data) # (B, hidden_dim)
        dart_encoder.train()
        dart_encoder.zero_grad()
        dart_emb = dart_encoder(**dart_data) # (B, hidden_dim)
        sim_score = torch.matmul( news_emb, dart_emb.T ) # (B , B)
                
            
        debug_idx = len(dataloader) // 10
        if batch_idx % debug_idx == 0 :
            log_print(f"corp_code : {corp_code}")
            log_print(f"dic : {dic}")
            log_print(f"sim_score[0] : {sim_score[0]}")
            log_print(f"loss so far {e_loss / (batch_idx+1)}")
            
        
        
            
            
        target = torch.arange(start=0, end = batch_size).to(device).to(torch.int64)
    
        CE_loss = CrossEntropyLoss()
        loss = CE_loss(sim_score, target)
        e_loss += loss.item()

        pred = torch.argmax(sim_score, dim = 1)
        for i, p in enumerate( pred ):
            if p == i:
                num_correct += 1
        
        
        loss.backward()
        dart_optimizer.step()
        news_optimizer.step()

        del dart_emb
        del news_emb
        torch.cuda.empty_cache()
    log_print(f"epoch : {e}, loss : { e_loss / len(dataloader) }")
    total_data_size =  len(dataloader) * BATCH_SIZE
    log_print(f"num_ccorect : {num_correct}, total_data_size : {total_data_size}, accuracy: { num_correct / total_data_size}")


# In[ ]:





# # validation

# In[43]:


# e_loss = 0
# news_encoder.eval()
# dart_encoder.eval()
# num_correct = 0
# for batch in tq.tqdm( val_dataloader ):
#     news_data = {
#         'input_ids' : batch[0].to(device).to(torch.int64), # ( B, seq_len )
#         'token_type_ids' : batch[1].to(device).to(torch.int64),
#         'attention_mask' : batch[2].to(device).to(torch.int64),
#     }
    
#     dart_data = {
#         'input_ids' : batch[3].to(device).to(torch.int64), # ( B, seq_len )
#         'token_type_ids' : batch[4].to(device).to(torch.int64),
#         'attention_mask' : batch[5].to(device).to(torch.int64),
#     }

#     corp_table = batch[6]


#     corp_code = corp_table.tolist()
#     dic = {} # for pooling for same corp


#     batch_size = batch[0].shape[0]
#     del batch
    
#     with torch.no_grad():
#         news_emb = news_encoder(**news_data) # (B, hidden_dim)
#         dart_emb = dart_encoder(**dart_data) # (B, hidden_dim)

#         sim_score = torch.matmul( news_emb, dart_emb.T ) # (B , B)

#         top_idx = torch.argmax(sim_score, dim = 1) # (B)

        
#         for p_i, pred in enumerate(top_idx):
#             if p_i == pred:
#                 num_correct += 1
        
#         target = torch.arange(start=0, end = batch_size).to(device).to(torch.int64)

#         CE_loss = CrossEntropyLoss()
#         loss = CE_loss(sim_score, target)
#         e_loss += loss.item()
# print(e_loss / len(val_dataloader))


# In[44]:


# print(num_correct / (len(val_dataloader) * BATCH_SIZE))

