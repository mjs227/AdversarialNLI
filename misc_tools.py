
import gc
import torch


LBL_DICT = {
    'LABEL_0': 'ENTAILMENT',
    'LABEL_1': 'NEUTRAL',
    'LABEL_2': 'CONTRADICTION',
    'ENTAILMENT': 'ENTAILMENT',
    'NEUTRAL': 'NEUTRAL',
    'CONTRADICTION': 'CONTRADICTION'
}
HF_MODEL_LIST = [
    'pepa/roberta-large-snli', 'pepa/deberta-v3-large-snli',
    'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
    'ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli',
    'roberta-large-mnli', 'facebook/bart-large-mnli'
]


def collect():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def eval_performance(hf_pipe, eval_ds, batch_size=None):
    update_fn = lambda z: {**z, **{'label': LBL_DICT[z['label'].upper()], }}
    eval_ds_fn = lambda w, z: {**w, **{'corr': w['label'] == z['label'].upper()}}

    with torch.no_grad():
        if batch_size is None:
            eval_res = hf_pipe([x['s1'] + ' ' + x['s2'] for x in eval_ds])
        else:
            eval_res = hf_pipe([x['s1'] + ' ' + x['s2'] for x in eval_ds], batch_size=batch_size)

        out_res = [eval_ds_fn(update_fn(eval_res[i_]), eval_ds[i_]) for i_ in range(len(eval_res))]

    return out_res, sum(x['corr'] for x in out_res) / len(eval_ds)
