from transformers import BertModel, AutoModelForTokenClassification
from tokenization_kobert import KoBertTokenizer
import os
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
import tqdm
import torch

model = AutoModelForTokenClassification.from_pretrained('monologg/kobert')
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
def convert_input_file_to_tensor_dataset(lines,                                         
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []
    max_seq_len = 100

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([0] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset

def get_labels(label_file):
    return [label.strip() for label in open(os.path.join(label_file), 'r', encoding='utf-8')]
def read_input_file(input_file):
    lines = []
    try : 
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                words = line.split()
                lines.append(words)
    except:
        temp = "부문  주요제품  CE 부문  TV, 모니터, 냉장고, 세탁기, 에어컨 등  IM 부문  HHP, 네트워크시스템, 컴퓨터 등  DS 부문  DRAM, NAND Flash, 모바일AP, 스마트폰용 OLED 패널 등   Harman 부문  디지털 콕핏(Digital Cockpit), 텔레매틱스(Telematics), 스피커 등"
        line = temp.strip()
        words = temp.split()
        lines.append(words)
    return lines



def predict():
    # load model and args    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained('monologg/kobert')  # Config will be automatically loaded from model_dir
    model.to(device)
    model.eval()    
    label_lst = get_labels("knowledge_graph/NER/label.txt")   
    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index    
    lines = read_input_file("knowledge_graph/NER/pred_test.txt")
    dataset = convert_input_file_to_tensor_dataset(lines, tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=32)

    all_slot_label_mask = None
    preds = None

    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(preds.shape[0])]

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                preds_list[i].append(slot_label_map[preds[i][j]])

    # Write to output file
    with open("ner_result.txt", "w", encoding="utf-8") as f:
        for words, preds in zip(lines, preds_list):
            line = ""
            for word, pred in zip(words, preds):
                if pred == 'O':
                    line = line + word + " "
                else:
                    line = line + "[{}:{}] ".format(word, pred)

            f.write("{}\n".format(line.strip()))
    

predict()