import os
import sys
import time
import random
import argparse
import datetime

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from transformers import AdamW
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import ElectraForSequenceClassification

from libagc.classification.classes import QType5

QType = QType5
#TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "../../weights/classification_tokenizer")
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# 시간 표시 함수
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

class ElectraClassifier():
    def __init__(self, phase='test'):
        self.phase = phase
        self.num_labels = len(QType)-1 # Exclude Unknown Type
        
        if phase == 'train':
            self.model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=self.num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = self.model.to(device)

    def classify(self, question):
        self.model.eval()
        token_res = self.tokenizer(question, padding="max_length", max_length=128, truncation=True)
        input_ids = torch.tensor([token_res['input_ids']]).to(device)
        mask = torch.tensor([token_res['attention_mask']]).to(device)

        outputs = self.model(input_ids, token_type_ids=None, attention_mask=mask).logits

        q_type_id = np.argmax(outputs.to('cpu').detach().numpy()[0])
        return QType(q_type_id)

    def __call__(self, question):
        return self.classify(question)

    def train(self):
        model = self.model
        
        # Hyper Parameters
        lr = 5e-5
        epochs = 20
        batch_size = 32

        # Prepre dataset
        train_dataset = QuestionDataset(classifier.tokenizer, is_train=True)
        test_dataset = QuestionDataset( classifier.tokenizer, is_train=False)
        sampler = WeightedRandomSampler(train_dataset.sample_weights, len(train_dataset.sample_weights))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=1, sampler=sampler
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=1, shuffle=False
        )

        optimizer = AdamW(model.parameters(),
                          lr = lr,
                          eps = 1e-8 )

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = 100,
                                                    num_training_steps = total_steps)

        model.zero_grad()
        # ========================================
        #               Train
        # ========================================
        for epoch_i in range(0, epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()
            total_loss = 0

            model.train()
            for step, batch in enumerate(train_loader):
                if step % 500 == 0 and not step == 0:
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                # Forward-pass
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

                loss = outputs[0]
                total_loss += loss.item()

                # Back-propagation
                loss.backward()
                # Gradient clips less than or equal to 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Gradient updates
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            avg_train_loss = total_loss / len(train_loader)

            print("")
            print("  Average training loss: {0:.6f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            t0 = time.time()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            model.eval()
            for batch in test_loader:
                
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                # Forward-pass without using no gradient
                with torch.no_grad():
                    outputs = model(b_input_ids,
                                    token_type_ids = None,
                                    attention_mask = b_input_mask)

                logits = outputs[0]

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            print("  Accuracy: {0:.6f}".format(eval_accuracy/nb_eval_steps))
            print("  Validation took: {:}".format(format_time(time.time() - t0)))
            model.save_pretrained(f"{WEIGHT_PATH}")

        print("")
        print("Training complete!")

def get_qtype_by_column(column):
    if column.startswith('산술') or column.startswith('도형') or column.startswith('순서정하기1') or column.startswith('수찾기3'):
        return QType.Arithmetic
    if column.startswith('조합하기'):
        return QType.Combination
    elif column.startswith('수찾기1'):
        return QType.FindingNumber1
    elif column.startswith('수찾기2'):
        return QType.FindingNumber2
    elif column.startswith('크기비교') or column.startswith('순서정하기2'):
        return QType.Comparison
    else:
        return QType.Unknown

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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type = str, default = 'test', choices = ['train', 'test'])
    parser.add_argument('--seed', type=int, default = 42)
    args = parser.parse_args()

    # best result when seed = 42
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    classifier = ElectraClassifier(phase = args.phase)
    if args.phase == 'train':
        classifier.train()

    q_set = ["바구니에 사과가 8개 있습니다. 정국이가 바구니에 사과 7개를 더 넣었습니다. 바구니 안에 있는 사과는 모두 몇 개입니까?",
            "유정이는 가지고 있던 사탕 중에서 언니에게 7개를 주고 동생에게 6개를 주 었더니 15개가 남았습니다. 처음에 유정이가 가지고 있던 사탕은 몇 개입니까?",
            "감과 귤이 모두 합해서 129개 있습니다. 감이 귤보다 43개 더 적다면 감은 몇 개 있습니까?",
            "색종이를 4명에게 똑같이 나누어 주어야 할 것을 잘못하여 5명에게 똑같이 나누어 주었더니 한 사람당 15장씩 주고 2장이 남았습니다. 이 색종이를 4명에게 똑같이 나누어 주면 한 사람당 최대한 몇 장씩 가지게 됩니까?",
            "윤기네 반 전체 학생 수는 32명입니다. 그중에서 남학생은 전체의 5/8입니다. 남학생 중에서 안경을 낀 학생은 남학생 전체의 3/4입니다. 윤기네 반에 서 안경을 끼지 않은 남학생은 몇 명입니까?",
            "학교에서 국어, 수학, 영어, 과학, 사회의 순서로 시험을 봤습니다. 3번째로 시험을 본 과목은 무엇입니까?",
            "달리기 시합에서 정국이는 7등을 했고, 민영이는 5등을 했습니다. 태형이는 민영이보다 못했지만 정국이보다는 잘했습니다. 태형이는 몇 등입니까?",
            "학생들이 한 줄로 서 있습니다. 유정이는 맨 뒤에 서 있습니다. 은정이는 앞에서 5번째에 서 있습니다. 은정이와 유정이 사이에 8명이 서 있을 때, 줄을 서 있는 학생은 모두 몇 명입니까?",
            "윤기는 왼쪽에서 7번째 열, 오른쪽에서 13번째 열, 앞에서 8번째 줄, 뒤에서 14번째 줄에 서서 체조를 하고 있습니다. 각 줄마다 서 있는 학생의 수가 같다고 할 때, 체조를 하고 있는 학생은 모두 몇 명입니까?",
            "도서관에 똑같은 책장이 28개 있습니다. 각 책장은 6층이고, 각 층마다 꽂혀 있는 책의 수는 같습니다. 영어책은 어느 책장의 한 층의 왼쪽에서 9번째, 오른쪽에서 11번째에 꽂혀 있습니다. 도서관의 책장에 꽂혀 있는 책은 모두 몇 권입니까?",
            "4, 2, 1 중에서 서로 다른 숫자 2개를 뽑아 만들 수 있는 가장 큰 두 자리 수를 구하시오.",
            "8, 3, 2, 6 중에서 서로 다른 숫자 2개를 뽑아 만들 수 있는 두 자리 수 중에서 가장 큰 수와 가장 작은 수의 차를 구하시오.",
            "1부터 30까지 자연수를 쓰려고 합니다. 숫자 2는 모두 몇 번 써야 합니까?",
            "0, 2, 4, 6, 8 중 서로 다른 2개의 숫자를 뽑아서 두 자리 수를 만들려고 합니다. 만들 수 있는 두 자리 수는 모두 몇 개입니까?",
            "10보다 작은 자연수 중에서 서로 다른 세 수를 동시에 뽑으려고 합니다. 세 수의 합이 12인 경우의 수를 구하시오.",
            "43, 92, 71, 64가 있습니다. 그중에서 가장 큰 수에서 가장 작은 수를 뺀 값 은 얼마입니까?",
            "0, 3, 5, 6 중에서 서로 다른 숫자 3개를 뽑아 만들 수 있는 세 자리 수 중에서 가장 작은 수를 쓰시오.",
            "5개의 수 1.4, 9/10, 1, 0.5, 13/10이 있습니다. 이 중에서 1보다 큰 수는 모두 몇 개입니까?",
            "유나가 책을 펼쳤는데 두 쪽수의 합이 125이었습니다. 유나가 펼친 두 쪽수 중 큰 수를 쓰시오.",
            "어떤 소수의 소수점을 왼쪽으로 두 자리 옮기면 원래의 소수보다 1.782만큼 작아집니다. 원래의 소수를 구하시오.",
            "A, B는 한 자리 수입니다. A는 2보다 4 큰 수이고, B보다 3 작은 수는 1입니다. A와 B의 합을 구하시오.",
            "서로 다른 두 수 A, B가 있습니다. 두 자리 수끼리의 뺄셈식 8A-B2=45에서 A와 B의 합을 구하시오.",
            "서로 다른 네 수 A, B, C, D가 있습니다. 세 자리 수끼리의 덧셈식 7A4+B6C=D29에서 D를 구하시오.",
            "서로 다른 두 자연수 A, B가 있습니다. A를 17로 나누면 몫은 25이고 나머지는 B가 됩니다. 나머지 B가 가장 큰 수일 때 A를 구하시오.",
            "네 자리 수 6A42를 백의 자리에서 반올림하면 6000이 됩니다. 0부터 9까지의 숫자 중 A에 쓸 수 있는 숫자는 모두 몇 개입니까?",
            "어떤 수에서 7을 뺐더니 2가 되었습니다. 어떤 수를 구하시오.",
            "어떤 수에서 46을 빼야 하는데 잘못하여 59를 뺐더니 43이 되었습니다. 바르게 계산한 결과를 구하시오.",
            "두 자리 수끼리의 곱셈에서 곱하는 수의 십의 자리 숫자 2를 6으로 잘못 보고 계산한 값이 2432가 되었습니다. 바르게 계산한 값이 912일 때, 2개의 두 자리 수 중 더 작은 수를 쓰시오.",
            "어떤 수에 14를 더한 후 14를 곱하고, 24를 뺀 값을 24로 나누면 13이 됩니다. 어떤 수를 구하시오.",
            "12에 어떤 수를 곱해야 하는데 잘못하여 어떤 수를 12로 나누었더니 8이 되었습니다. 바르게 계산한 결과를 구하시오.",
            "정국이는 지우개를 6개 가지고 있습니다. 지민이는 지우개를 정국이보다 4 개 더 많이 가지고 있고 석진이는 지민이보다 3개 더 적게 가지고 있습니 다. 지우개를 가장 적게 가지고 있는 사람은 누구입니까?",
            "네 수 A, B, C, D가 있습니다. A는 27입니다. B는 A보다 7 큰 수입니다. C는 B보다 9 작은 수입니다. D는 C의 2배인 수입니다. 가장 큰 수는 어느 것입니까?",
            "석진이는 호석이보다 무겁고 지민이보다 가볍습니다. 남준이는 지민이보다 무겁습니다. 4명 중 가장 가벼운 사람은 누구입니까?",
            "지민이는 주스를 0.7l 마셨습니다. 은지는 지민이보다 1/10l 더 적게 마셨습니다. 윤기는 4/5l 마셨고, 유나는 지민이보다 0.2l 더 많이 마셨습니다. 주스를 가장 많이 마신 사람은 누구입니까?",
            "철수, 영수, 영철, 경수, 경환 5명이 있습니다. 철수는 나이가 가장 적습니다. 영수는 경수에게는 동생이고 경환에게는 형입니다. 경수는 2년 후에 40살이 되고, 영철이는 올해 40살입니다. 5명 중에서 나이가 2번째로 적은 사람은 누구입니까?",
            "삼각형의 변의 개수와 사각형의 변의 개수의 합을 구하시오.",
            "한 변의 길이가 5cm인 정삼각형의 둘레는 몇 cm입니까?",
            "길이가 20㎝인 철사로 직사각형을 만들었더니 철사가 남지도 모자라지도 않았습니다. 직사각형의 가로 길이가 4cm일 때, 세로 길이는 몇 cm입니까?",
            "한 변의 길이가 10cm인 정사각형과 둘레가 같은 정팔각형이 있습니다. 이 정팔각형의 한 변의 길이는 몇 cm입니까?",
            "둘레가 24cm인 직사각형이 있습니다. 이 직사각형의 가로 길이가 세로 길이의 2배일 때 가로는 몇 cm입니까?"]
    
    a_set = [0,0,0,0,0,
             4,0,0,0,0,
             1,1,1,1,1,
             2,1,2,0,0,
             3,3,3,3,3,
             0,0,0,0,0,
             4,4,4,4,4,
             0,0,0,0,0]

    class_to_num = {QType.Arithmetic: 0,
                    QType.Combination: 1,
                    QType.FindingNumber1: 2,
                    QType.FindingNumber2: 3,
                    QType.Comparison: 4}

    correct = 0
    for i, (q, a) in enumerate(zip(q_set, a_set)):
        print(f"{i+1}번 문제: {q}")
        p = class_to_num.get(classifier.classify(q))
        print(f"추론/정답 카테고리: {p}/{a}\n")
        if p == a:
            correct += 1
    print(f"문제 수: {len(q_set)}")
    print(f"맞은 문제 수: {correct}")
    print(f"틀린 문제 수: {len(q_set) - correct}")
    print(f"정답률 {correct/len(q_set):.3f}")
