from haven import haven_utils as hu
from args import parse_args
import numpy as np
import json
import os

CONFIG_FOLDER = 'Config/Best_params'

EXP_GROUPS = {}
EXP_GROUPS['single_final_result'] = []

'''
This file will reproduce a single result of the experiments in the paper

choose and dataset, non-stationarity level and model_name_impv

model_name_impv = {'online_sgd', 'fine_tuning', 'MetaCOG', 'MetaBGD', 'MAML','ANIL', 'BGD', 
                    'ours', 'ours_pre', 'ours_kwto', 'ours_pre_kwto', 'ours_kwto_acc',
                    'ours_pre_kwto_acc'}

ours = c-maml
+ pre = pretraining
+ kwto = update modulation (UM)
+ acc = prolonged adaptation phase (PAP)

'''


experiment = 'omniglot' # 'synbols', 'tiered-imagenet'
prob_statio = 0.98 # 0.9
model_name_impv = 'ours' 

boilerplate_final = {
    "verbose":1,
    #wandb":"args.wandb"
    "folder":"Data",
    "n_runs":1,
    "dataset": experiment,
}

params = []
raw_params = json.load(open(os.path.join(CONFIG_FOLDER,"{}_p{}.json".format(experiment, prob_statio))))

specified_model_params = next(item for item in raw_params if item["model_name_impv"] == "ours")

specified_model_params = [dict(specified_model_params, **boilerplate_final)]

EXP_GROUPS['single_final_result'] += specified_model_params


