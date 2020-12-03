from haven import haven_utils as hu
from args import parse_args, postprocess_args
import numpy as np
import json
import os

CONFIG_FOLDER = 'Config/Best_params'

EXP_GROUPS = {}
EXP_GROUPS['final_results'] = []

for experiment in ['omniglot', 'synbols', 'tiered-imagenet']:

    boilerplate_final = {
        "verbose":1,
        #wandb":"args.wandb"
        "folder":"Data",
        "n_runs":20,
        "dataset": experiment,
    }

    params = []
    for prob_statio in [0.90, 0.98]:
        raw_params = json.load(open(os.path.join(CONFIG_FOLDER,"{}_p{}.json".format(experiment, prob_statio))))
        raw_params = [dict(raw_params[i], **boilerplate_final) for i in range(len(raw_params))]

        params += raw_params


    EXP_GROUPS['final_results'] += params


