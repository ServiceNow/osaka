import numpy as np 
from os import system

'''
Hyperparameter sweep
See table 7 of the paper https://arxiv.org/pdf/2003.05856.pdf
Using EAI's toolkit
'''

TOOLKIT=True #for using EAI's Toolkit or not 
N_TRIALS = 300
WANDB = 'osaka_omniglot_sparseMAML'
WANDB_KEY = '9f084a9bf4cb3d531e38838a19bd8480216da061'

#-------
#setting
DATASET = ['omniglot'] # 'synbols', 'tiered_imagenet']
PROB_STATIO = [0.9, 0.98]

#-------
# method
# MODEL_NAME = ['ANIL', 'BGD', 'fine_tuning', 'MAML', 'MetaBGD', 'MetaCOG', 'online_sgd', 'ours', 'sparseMAML']
MODEL_NAME = ['sparseMAML', 'ours', 'MAML']

# LEARN_STEP_SIZE = [True, False]
LEARN_STEP_SIZE = [True]
#NOTE: we will automatically set per_parameter_step_size = learn_step_size

META_LR = [0.0001, 0.0005, 0.005, 0.001, 0.05, 0.01] # $\eta$ 
BATCH_SIZE = [1, 2, 4, 8, 16]
INNER_STEP_SIZE = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5] 
NUM_STEPS = [1, 2, 4, 8, 16]
FIRST_ORDER =[False, True]

# BGD (BDG, metaBDG, metaCOG)
MEAN_ETA = [0.5, 1.0, 10.] # $\beta$
STD_INIT = [0.001, 0.01, 0.1] # $\sigma$

# C-MAML 
#NOTE: for cl_strategy_thres=0: C-MAML + UM collapses to MAML
# for cl_strategy=1000: C-MAML + UM collapses to C-MAML 
CL_STRATEGY_THRES = [0, 0.25, 0.5, 1, 2, 3, 5, 1000] # $\lambda$
#NOTE: for cl_tbd_thres: C-MAML will always retrain on incomin data
CL_TBD_THRES = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 1000] # $\gamma$
CL_ACCUMULATE = [True, False]


# Pretraining (fine-tuning, MAML, ANIL, C-MAML + Pre) 
# set NUM_EPOCHS to 0 for none, or to a big value for pre-training (there is early stopping) 
NUM_EPOCHS = [0, 0, 1000]
PATIENCE = [5, 10, 20] 

# metaCOG
MASKS_INIT = [0.1, 0.2, 0.3, 0.4, 0.5]
L1_REG = [0, 0.1, 1, 10]

# sparseMAML
STEP_SIZE_ACTIVATION = [None, 'binary_trough', 'relu_trough']

#------
# sweep

for _ in range(N_TRIALS):

    dataset = np.random.choice(DATASET)
    prob_statio = np.random.choice(PROB_STATIO)
    
    model_name = np.random.choice(MODEL_NAME)
    learn_step_size = np.random.choice(LEARN_STEP_SIZE)
    per_param_step_size = learn_step_size
    meta_lr = np.random.choice(META_LR)
    batch_size = np.random.choice(BATCH_SIZE)
    step_size = np.random.choice(INNER_STEP_SIZE)
    num_steps = np.random.choice(NUM_STEPS)
    first_order = np.random.choice(FIRST_ORDER)
    mean_eta = np.random.choice(MEAN_ETA)
    std_init = np.random.choice(STD_INIT)
    cl_strategy_thres = np.random.choice(CL_STRATEGY_THRES)
    cl_tbd_thres = np.random.choice(CL_TBD_THRES)
    cl_accumulate = np.random.choice(CL_ACCUMULATE)
    num_epochs = np.random.choice(NUM_EPOCHS)
    patience = np.random.choice(PATIENCE)
    masks_init = np.random.choice(MASKS_INIT)
    l1_reg = np.random.choice(L1_REG)
    step_size_activation = np.random.choice(STEP_SIZE_ACTIVATION)
    
    #-----------
    # launch job
    command = f'--wandb {WANDB} ' \
              f'--wandb_key {WANDB_KEY} ' \
              f'--dataset {dataset} ' \
              f'--prob_statio {prob_statio} ' \
              f'--model_name {model_name} ' \
              f'--learn_step_size {learn_step_size} ' \
              f'--per_param_step_size {per_param_step_size} ' \
              f'--meta_lr {meta_lr} ' \
              f'--batch_size {batch_size} ' \
              f'--step_size {step_size} ' \
              f'--num_steps {num_steps} ' \
              f'--first_order {first_order} ' \
              f'--mean_eta {mean_eta} ' \
              f'--cl_strategy_thres {cl_strategy_thres} ' \
              f'--cl_tbd_thres {cl_tbd_thres} ' \
              f'--cl_accumulate {cl_accumulate} ' \
              f'--num_epochs {num_epochs} ' \
              f'--patience {patience} ' \
              f'--masks_init {masks_init} ' \
              f'--l1_reg {l1_reg} ' \
              f'--step_size_activation {step_size_activation} ' 


    print(command)
    if TOOLKIT:             
        system(f'bash toolkit_job2.sh {command} "$@" ')
    else:
        system(f'python main.py {command}')

