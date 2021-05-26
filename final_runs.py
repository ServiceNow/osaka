import numpy as np 
from os import system

'''
Hyperparameter sweep
See table 7 of the paper https://arxiv.org/pdf/2003.05856.pdf
Using EAI's toolkit
'''

TOOLKIT=True #for using EAI's Toolkit or not 
WANDB = 'osaka_omniglot_sparseMAML'
WANDB_KEY = '<insert key>'

RUNS = {
    'final_MAML_0.9': ' --wandb osaka_omniglot_sparseMAML --wandb_key 9f084a9bf4cb3d531e38838a19bd8480216da061 --dataset omniglot --prob_statio 0.9 --model_name MAML --learn_step_size True --per_param_step_size True --meta_lr 0.001 --batch_size 1 --step_size 0.001 --num_steps 8 --first_order True --mean_eta 0.5 --cl_strategy_thres 1000.0 --cl_tbd_thres 3.0 --cl_accumulate False --num_epochs 0 --patience 10 --masks_init 0.4 --l1_reg 10.0 --step_size_activation None',
    'final_MAML_sparse_0.9': ' --wandb osaka_omniglot_sparseMAML --wandb_key 9f084a9bf4cb3d531e38838a19bd8480216da061 --dataset omniglot --prob_statio 0.9 --model_name MAML --learn_step_size True --per_param_step_size True --meta_lr 0.0005 --batch_size 4 --step_size 0.1 --num_steps 16 --first_order False --mean_eta 0.5 --cl_strategy_thres 2.0 --cl_tbd_thres 0.5 --cl_accumulate False --num_epochs 1000 --patience 5 --masks_init 0.5 --l1_reg 0.0 --step_size_activation relu_trough',
    'final_MAML_0.98': ' --wandb osaka_omniglot_sparseMAML --wandb_key 9f084a9bf4cb3d531e38838a19bd8480216da061 --dataset omniglot --prob_statio 0.98 --model_name MAML --learn_step_size True --per_param_step_size True --meta_lr 0.0001 --batch_size 16 --step_size 0.0005 --num_steps 16 --first_order False --mean_eta 10.0 --cl_strategy_thres 2.0 --cl_tbd_thres 3.0 --cl_accumulate False --num_epochs 0 --patience 5 --masks_init 0.1 --l1_reg 10.0 --step_size_activation None',
    'final_MAML_sparse_0.98': ' --wandb osaka_omniglot_sparseMAML --wandb_key 9f084a9bf4cb3d531e38838a19bd8480216da061 --dataset omniglot --prob_statio 0.98 --model_name MAML --learn_step_size True --per_param_step_size True --meta_lr 0.001 --batch_size 8 --step_size 0.05 --num_steps 16 --first_order True --mean_eta 0.5 --cl_strategy_thres 5.0 --cl_tbd_thres 0.75 --cl_accumulate False --num_epochs 0 --patience 10 --masks_init 0.4 --l1_reg 0.1 --step_size_activation relu_trough',
    'final_CMAML_0.9': ' --wandb osaka_omniglot_sparseMAML --wandb_key 9f084a9bf4cb3d531e38838a19bd8480216da061 --dataset omniglot --prob_statio 0.9 --model_name sparseMAML --learn_step_size True --per_param_step_size True --meta_lr 0.001 --batch_size 8 --step_size 0.5 --num_steps 4 --first_order True --mean_eta 10.0 --cl_strategy_thres 0.0 --cl_tbd_thres 1.0 --cl_accumulate False --num_epochs 0 --patience 20 --masks_init 0.4 --l1_reg 10.0 --step_size_activation None',
    'final_CMAML_sparse_0.9':' --wandb osaka_omniglot_sparseMAML --wandb_key 9f084a9bf4cb3d531e38838a19bd8480216da061 --dataset omniglot --prob_statio 0.9 --model_name sparseMAML --learn_step_size False --per_param_step_size False --meta_lr 0.001 --batch_size 8 --step_size 0.1 --num_steps 8 --first_order True --mean_eta 0.5 --cl_strategy_thres 0.5 --cl_tbd_thres 1.5 --cl_accumulate True --num_epochs 0 --patience 20 --masks_init 0.1 --l1_reg 1.0 --step_size_activation relu_trough',
    'final_CMAML_0.98':' --wandb osaka_omniglot_sparseMAML --wandb_key 9f084a9bf4cb3d531e38838a19bd8480216da061 --dataset omniglot --prob_statio 0.98 --model_name sparseMAML --learn_step_size True --per_param_step_size True --meta_lr 0.01 --batch_size 1 --step_size 0.001 --num_steps 4 --first_order False --mean_eta 0 --cl_strategy_thres 3.0 --cl_tbd_thres 0.5 --cl_accumulate False --num_epochs 0 --patience 10 --masks_init 0.5 --l1_reg 10.0 --step_size_activation None',
    'final_CMAML_sparse_0.98':' --wandb osaka_omniglot_sparseMAML --wandb_key 9f084a9bf4cb3d531e38838a19bd8480216da061 --dataset omniglot --prob_statio 0.98 --model_name sparseMAML --learn_step_size False --per_param_step_size False --meta_lr 0.0005 --batch_size 4 --step_size 0.005 --num_steps 16 --first_order True --mean_eta 1.0 --cl_strategy_thres 1.0 --cl_tbd_thres 2.0 --cl_accumulate True --num_epochs 0 --patience 10 --masks_init 0.4 --l1_reg 0.0 --step_size_activation relu_trough',
}

for run_name, command in RUNS.items():

    command += f' --name {run_name} --n_runs 20'


    print(command)
    if TOOLKIT:             
        system(f'bash toolkit_job2.sh {command} "$@" ')
    else:
        system(f'python main.py {command}')

