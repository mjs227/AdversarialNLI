
import os
import json
import torch
import misc_tools
from transformers import pipeline


INOC_N = 1  # number of ext. neg. prefixes the models were inoculated on (state dicts to load)
NEG_TYPE = 'nt'  # 'nt' => "it is not true that"; 'f' => "it is false that"
EVAL_RANGE = (2, 5)  # range of ext. neg. prefixes to eval. models on (EVAL_RANGE[0] should be > INOC_N)
EVAL_BATCH_SIZE = 128
FILEPATH = '/path/to/files/'

EVAL_BATCH_SIZE, _DEVICE = (EVAL_BATCH_SIZE, 0) if torch.cuda.is_available() else (None, 'cpu')
FILEPATH = (os.path.abspath(FILEPATH) + '/').replace('//', '/')
assert NEG_TYPE in {'nt', 'f'}

for n in range(EVAL_RANGE[0], EVAL_RANGE[1] + 1):
    print(f'Loading depth-{n} MNLI data...')

    with open(f'{FILEPATH}mnli_adv_{n}{NEG_TYPE}.json', 'r') as f:
        mnli_data = json.load(f)

    print(f'Loading depth-{n} SNLI data...')

    with open(f'{FILEPATH}snli_adv_{n}{NEG_TYPE}.json', 'r') as f:
        snli_data = json.load(f)

    for hf_model in misc_tools.HF_MODEL_LIST:
        if hf_model[:4] == 'pepa':
            ds_name, ds_file = 'snli', snli_data
        elif hf_model[-4:] == 'mnli':
            ds_name, ds_file = 'mnli', mnli_data
        else:
            ds_name, ds_file = 'both', mnli_data + snli_data

        print(f'\n{hf_model} ({ds_name}):')

        model_save_name = hf_model.replace('/', '_')
        pipe = pipeline('text-classification', model=hf_model, device=_DEVICE)
        pipe.model.eval()

        init_res_list, init_acc = misc_tools.eval_performance(pipe, ds_file, batch_size=EVAL_BATCH_SIZE)
        misc_tools.collect()

        print(f'Initial depth-{INOC_N} accuracy: {init_acc}')

        pipe.model.to('cpu')
        pipe.model.load_state_dict(torch.load(f'{FILEPATH}depth_{INOC_N}_inoc/state_dicts/{model_save_name}.pt'))
        pipe.model.to(_DEVICE)

        inoc_res_list, inoc_acc = misc_tools.eval_performance(pipe, ds_file, batch_size=EVAL_BATCH_SIZE)

        print(f'Inoculated depth-{INOC_N} accuracy: {inoc_acc}')

        out_json = {
            'init': {
                'results': init_res_list,
                'acc': init_acc
            },
            'inoc': {
                'results': inoc_res_list,
                'acc': inoc_acc
            }
        }

        with open(f'{FILEPATH}depth_{INOC_N}/evaluation_res/{n}{NEG_TYPE}_{model_save_name}.json', 'w') as f:
            json.dump(out_json, f)

        del pipe
        del init_res_list
        del inoc_res_list
        misc_tools.collect()
