
import os
import json
import torch
import spacy
import random


DATASET = 'SNLI'
NUM_NEG = 8  # number of times to repeat external negation prefix
MULTI_NEG = False  # True => create D^{<=NUM_NEG}; False => create D_{NT/F}^{NUM_NEG}
CREATE_DEV = True  # create dev set for inoc. (ONLY original examples---no examples in DEV have ext. neg. prefix)
CREATE_ADV = True  # create challenge test set (ALL examples in ADV have ext. neg. prefix)
CREATE_TRAIN = True  # create inoc. train set (ALL examples in TRAIN have ext. neg. prefix)
NEG_TYPE = 'nt'  # 'nt' => "it is not true that"; 'f' => "it is false that"
FILEPATH = '/path/to/files/'

_LBL_DICT = {
    0: {
        'CONTRADICTION': 'CONTRADICTION',
        'NEUTRAL': 'NEUTRAL',
        'ENTAILMENT': 'ENTAILMENT'
    },
    1: {
        'CONTRADICTION': 'ENTAILMENT',
        'NEUTRAL': 'NEUTRAL',
        'ENTAILMENT': 'CONTRADICTION'
    }
}
_NER_MODEL = spacy.load('en_core_web_sm')


def decap_first_w(s):
    doc, min_idx = _NER_MODEL(s), 1

    if len(doc.ents) > 0:
        min_idx = min(ent.start_char for ent in doc.ents)

    if min_idx == 0 or (s[0].lower() == 'i' and not s[1].isalnum()):
        return s[0].upper() + s[1:]

    return s[0].lower() + s[1:]


def get_neg_fn(fn_type):
    if fn_type == 'nt':
        trigger_phrase = 'it is not true that '
    elif fn_type == 'f':
        trigger_phrase = 'it is false that '
    else:
        raise NotImplementedError

    def neg_(s, n=1):
        trigger = trigger_phrase * n

        return 'I' + trigger[1:] + decap_first_w(s)

    return neg_


NEG_TYPE = NEG_TYPE.strip().lower()
assert NEG_TYPE in {'nt', 'f'}
assert isinstance(NUM_NEG, int) and NUM_NEG > 0
assert CREATE_DEV or CREATE_ADV or CREATE_TRAIN
FILEPATH = (os.path.abspath(FILEPATH) + '/').replace('//', '/')
torch.manual_seed(1)
random.seed(1)

if DATASET == 'MNLI':
    dev_file_name = 'mnli_dev_matched.tsv'
    _S1_IDX, _S2_IDX, _LBL_IDX, _TREE_IDX = 8, 9, -1, 7
elif DATASET == 'SNLI':
    dev_file_name = 'snli_1.0_test.txt'
    _S1_IDX, _S2_IDX, _LBL_IDX, _TREE_IDX = 5, 6, 0, 4
else:
    raise NotImplementedError

neg = get_neg_fn(NEG_TYPE)
train_file_out = {k: [] for k in _LBL_DICT.keys()}
adv_file_out = {k: [] for k in _LBL_DICT.keys()}
dev_file_out = []

with open(FILEPATH + dev_file_name, 'r') as f:
    dev_file = f.readlines()

if CREATE_DEV:
    for items in map(lambda x: x.split('\t'), dev_file[1:]):
        s1 = items[_S1_IDX].strip()
        s2 = items[_S2_IDX].strip()
        label = items[_LBL_IDX].strip().upper()

        if s1[-1] not in {'.', '!', '?'}:
            s1 += '.'

        dev_file_out.append({'s1': s1, 's2': s2[0].upper() + s2[1:], 'label': label})

if CREATE_ADV or CREATE_TRAIN:
    with open(FILEPATH + DATASET.lower() + '_train.txt', 'r') as f:
        train_file = f.readlines()

    len_train, i = len(train_file) - 1, -1
    train_file_indices = torch.randperm(len_train).tolist()
    lbl_cnt_train = {k: 0 for k in _LBL_DICT.keys()}
    lbl_cnt_adv = {k: 0 for k in _LBL_DICT.keys()}
    lbl_lim_train = (len(dev_file) // 3) if CREATE_TRAIN else 0
    lbl_lim_adv = (len(dev_file) // 6) if CREATE_ADV else 0
    max_size, size_cnt = (lbl_lim_train + lbl_lim_adv) * 3, 0

    while size_cnt < max_size:
        i += 1
        items = train_file[train_file_indices[i] + 1].split('\t')
        label = items[_LBL_IDX].strip().upper()
        s2 = items[_S2_IDX].strip()

        if label in lbl_cnt_train.keys() and items[_TREE_IDX].strip()[:9] == '(ROOT (S ' and '?' not in s2:
            if lbl_cnt_train[label] < lbl_lim_train:
                lbl_cnt_train[label] += 1
                out_file = train_file_out
            elif lbl_cnt_adv[label] < lbl_lim_adv:
                lbl_cnt_adv[label] += 1
                out_file = adv_file_out
            else:
                continue

            size_cnt += 1
            s1 = items[_S1_IDX].strip()
            out_file[label].append({
                's1': s1 + ('' if s1[-1] in {'.', '!', '?'} else '.'),
                's2': s2,
                'index': train_file_indices[i] + 1,
            })

for out_file in [train_file_out, adv_file_out]:
    len_file = len(out_file['NEUTRAL'])
    assert sum(len(x) == len_file for _, x in out_file.items()) == 3

    if len_file > 0:  # i.e. if CREATE_[SPLIT] = True
        for lbl, lbl_list in out_file.items():
            random.shuffle(lbl_list)

            if MULTI_NEG:
                split_size = int(len_file // NUM_NEG)
                total_coverage = split_size * NUM_NEG
                remainder = len_file - total_coverage
                num_neg_fn = lambda z: z + 1
            else:
                total_coverage, remainder, split_size = 0, 0, len_file
                num_neg_fn = lambda z: NUM_NEG

            for i in range(NUM_NEG if MULTI_NEG else 1):
                num_neg = num_neg_fn(i)
                lbl_dict = _LBL_DICT[num_neg % 2]

                for j in range(i * split_size, (i + 1) * split_size):
                    lbl_list[j].update({
                        's2': neg(lbl_list[j]['s2'], num_neg),
                        'label': lbl_dict[lbl]
                    })

            if remainder > 0:
                for i in range(remainder):
                    num_neg = i % NUM_NEG
                    lbl_list[total_coverage + i].update({
                        's2': neg(lbl_list[total_coverage + i]['s2'], num_neg),
                        'label': _LBL_DICT[num_neg % 2][lbl]
                    })

train_file_out = sum((x for _, x in train_file_out.items()), [])
adv_file_out = sum((x for _, x in adv_file_out.items()), [])
random.shuffle(train_file_out)
random.shuffle(adv_file_out)
multi = '_MULTI' if MULTI_NEG else ''

for split, out_list in [('adv', adv_file_out), ('train', train_file_out)]:
    if len(out_list) > 0:
        with open(f'{FILEPATH}{DATASET.lower()}_{split}_{NUM_NEG}{NEG_TYPE}{multi}.json', 'w') as f:  # TODO
            json.dump(out_list, f)

if len(dev_file_out) > 0:
    with open(f'{FILEPATH}{DATASET.lower()}_dev.json', 'w') as f:
        json.dump(dev_file_out, f)
