import os
import sys
import datetime

import numpy as np
import pandas as pd

from enum import Enum

import torch
import json

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

class QType8(Enum):
    Arithmetic = 0
    Ordering = 1
    Combination = 2
    FindingNumber1 = 3
    FindingNumber2 = 4
    FindingNumber3 = 5
    Comparison = 6
    Geometry = 7

class QuestionDataset(Dataset):
    def __init__(self, tokenizer, is_train):
        self.tokenizer = tokenizer

        header=['new_cate', 'pre_cate', 'question', 'answer', 'formula1', 'formula2', 'candidate']
        excel_path = os.path.join(DATASET_PATH, "jenti_v1/eunyoung_v4_merge.xlsx")
        excel = pd.read_excel(excel_path, usecols=range(len(header)), names=header)

        q_dict = {}
        categories = excel['new_cate'].unique()
        for cate in categories:
            q_type = get_qtype_by_column(cate)
            if q_type == QType.Unknown:
                continue
            q = excel[excel['new_cate'] == cate]['question']
            q_type_id = q_type.value
            if q_type_id in q_dict:     # 다른 category, 동일한 type 대응 (6 class)
                q_dict[q_type_id].extend(q.to_list())
            else:
                q_dict[q_type_id] = q.to_list()

        train_dataset = {}
        val_dataset = {}
        for q_type_id in q_dict:
            all = q_dict[q_type_id]
        
            random.shuffle(all)

            # 10% of each category 
            all_slice = int(len(all)*0.1)
            # allocate 90% for training and the other 10% for test.
            # as seed number is fixed, they don't intersect. 
            train_dataset[q_type_id] = all[:-all_slice]
            val_dataset[q_type_id] = all[-all_slice:]
            
        self.all_data = []
        if is_train:
            self.sample_weights = []
            for q_type_id in train_dataset:
                count = len(train_dataset[q_type_id])
                for q in train_dataset[q_type_id]:
                    self.all_data.append((q, q_type_id))
                    self.sample_weights.append(1.0 / count)
        else:
            for q_type_id in val_dataset:
                for q in val_dataset[q_type_id]:
                    self.all_data.append((q, q_type_id))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        q, label = self.all_data[idx]
        token_res = self.tokenizer(q, padding="max_length", max_length=128, truncation=True)
        res = torch.tensor(token_res['input_ids']), torch.tensor(token_res['attention_mask']), torch.tensor(label)
        return res