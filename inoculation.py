
import os
import json
import torch
import misc_tools
from tqdm import tqdm
from copy import deepcopy
from transformers import pipeline


EVAL_BATCH_SIZE = 128
TRAIN_BATCH_SIZE = 32
LEARN_RATE = 1e-5
SAVE_PARAMS = False  # True => save best state dict; False => save last state dict
NUM_NEG = 1  # train/adv datasets (number of repeated external negation prefixes)
FILEPATH = '/path/to/files/'

_TENSOR_DICT = {'CONTRADICTION': [0, 0, 1], 'ENTAILMENT': [1, 0, 0], 'NEUTRAL': [0, 1, 0]}
_DEVICE = 0 if torch.cuda.is_available() else 'cpu'
_FLIP_TEN = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float).to(_DEVICE)
_ID_TEN = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float).to(_DEVICE)

torch.manual_seed(1)
torch.cuda.manual_seed(1)
FILEPATH = (os.path.abspath(FILEPATH) + '/').replace('//', '/')

mnli_train, snli_train = [], []
mnli_adv, snli_adv = [], []
mnli_dev, snli_dev = [], []
mnli_lbl, snli_lbl = [], []

for n, t, a, d, l in [('m', mnli_train, mnli_adv, mnli_dev, mnli_lbl), ('s', snli_train, snli_adv, snli_dev, snli_lbl)]:
    print(f'Loading {n.upper()}NLI data...')

    with open(f'{FILEPATH}{n}nli_train_{NUM_NEG}nt_MULTI.json', 'r') as f:
        train_file = json.load(f)

    with open(f'{FILEPATH}{n}nli_dev.json', 'r') as f:
        dev_file = json.load(f)

    with open(f'{FILEPATH}{n}nli_adv_{NUM_NEG}nt_MULTI.json', 'r') as f:
        adv_file = json.load(f)

    t += [x['s1'] + ' ' + x['s2'] for x in train_file]
    l += [_TENSOR_DICT[x['label']] for x in train_file]
    a += adv_file
    d += dev_file

mnli_lbl, snli_lbl = tuple(map(lambda x: torch.tensor(x, dtype=torch.float), (mnli_lbl, snli_lbl)))
both_train = mnli_train + snli_train
both_dev = mnli_dev + snli_dev
both_adv = mnli_adv + snli_adv
both_lbl = torch.cat((mnli_lbl, snli_lbl), dim=0)

DATASETS = {
    'mnli': (mnli_train, mnli_adv, mnli_dev, mnli_lbl, _FLIP_TEN),
    'snli': (snli_train, snli_adv, snli_dev, snli_lbl, _ID_TEN),
    'both': (both_train, both_adv, both_dev, both_lbl, _ID_TEN)
}

for hf_model in misc_tools.HF_MODEL_LIST:
    ds_name = 'snli' if hf_model[:4] == 'pepa' else ('mnli' if hf_model[-4:] == 'mnli' else 'both')

    print(f'\n{hf_model} ({ds_name}):')

    train_set, adv_set, dev_set, labels, trnsfrm = DATASETS[ds_name]
    pipe = pipeline('text-classification', model=hf_model, device=_DEVICE)
    model, tokenizer = pipe.model, pipe.tokenizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    optimizer.zero_grad()
    x_train = tokenizer(train_set, padding=True, truncation=True, return_tensors='pt').to(_DEVICE)
    x_train_ids, x_train_attn_mask = x_train['input_ids'], x_train['attention_mask']
    y_train = labels.to(_DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_state_dict = None

    num_batch, patience_cnt = -(-len(train_set) // TRAIN_BATCH_SIZE), 0
    train_stats = {'acc': [], 'loss': []}
    best_dev_res = []

    init_dev_res, prev_best_acc = misc_tools.eval_performance(pipe, dev_set, batch_size=EVAL_BATCH_SIZE)

    print(f'Initial dev-set accuracy: {prev_best_acc}')

    init_adv_res, init_adv_acc = misc_tools.eval_performance(pipe, adv_set, batch_size=EVAL_BATCH_SIZE)
    print(f'Initial challenge set accuracy: {init_adv_acc}')

    while patience_cnt < 5:
        print()
        print(f'Iteration {len(train_stats["acc"]) + 1}:')
        print()

        permutation, iter_loss = torch.randperm(len(train_set)).tolist(), 0
        model.train()

        for i in tqdm(range(num_batch)):
            batch_indices = permutation[i * TRAIN_BATCH_SIZE:(i + 1) * TRAIN_BATCH_SIZE]
            x_i_attn_mask = x_train_attn_mask[batch_indices]
            x_i_ids = x_train_ids[batch_indices]
            y_i = torch.matmul(model(input_ids=x_i_ids, attention_mask=x_i_attn_mask)['logits'], trnsfrm)
            loss = loss_fn(torch.softmax(y_i, dim=1), y_train[batch_indices])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter_loss += loss.item() * len(batch_indices)

            del x_i_ids, x_i_attn_mask, y_i
            misc_tools.collect()

        model.eval()
        iter_res, curr_acc = misc_tools.eval_performance(pipe, dev_set, batch_size=EVAL_BATCH_SIZE)

        if curr_acc > prev_best_acc:
            patience_cnt, prev_best_acc = 0, curr_acc

            if SAVE_PARAMS:
                model.to('cpu')
                best_state_dict = deepcopy(model.state_dict())
                best_dev_res = deepcopy(iter_res)
                model.to(_DEVICE)
        else:
            patience_cnt += 1

        train_stats['loss'].append(iter_loss / len(train_set))
        train_stats['acc'].append(curr_acc)

        print(f'Acc={curr_acc}, Best={prev_best_acc}, Patience={patience_cnt}')

        misc_tools.collect()

    print()
    print('\nInoculation complete!\n')

    if SAVE_PARAMS:
        model.to('cpu')
        model.load_state_dict(best_state_dict)
        model.to(_DEVICE)

    inoc_adv_res, inoc_adv_acc = misc_tools.eval_performance(pipe, adv_set, batch_size=EVAL_BATCH_SIZE)
    model_save_name = hf_model.replace('/', '_')

    print(f'Inoculated challenge set accuracy: {inoc_adv_acc}')

    with open(f'{FILEPATH}depth_{NUM_NEG}_inoc/inoculation_res/{model_save_name}.json', 'w') as f:
        json.dump({
            'init': {
                'dev': init_dev_res,
                'adv': init_adv_res
            },
            'inoc': {
                'dev': best_dev_res,
                'adv': inoc_adv_res
            },
            'stats': train_stats
        }, f)

    model.to('cpu')
    state_dict = model.state_dict()
    torch.save(state_dict, f'{FILEPATH}depth_{NUM_NEG}_inoc/state_dicts/{model_save_name}.pt')

    del pipe, model, tokenizer, x_train, y_train, optimizer
    del init_dev_res, init_adv_res
    del best_dev_res, inoc_adv_res
    del train_stats, best_state_dict
    misc_tools.collect()
