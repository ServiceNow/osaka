import torch
import torch.nn.functional as F

import math
import os
import time
import json
import argparse
import random
import logging
import numpy as np
import copy
from Exp_configs import EXP_GROUPS
import pprint

from pdb import set_trace

from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Compose
from dataloaders import init_dataloaders

from MAML.model import ModelConvSynbols, ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
from MAML.metalearners import ModelAgnosticMetaLearning, ModularMAML
from MAML.utils import ToTensor1D, set_seed, is_connected

from Utils.bgd_lib.bgd_optimizer import create_BGD_optimizer
from haven import haven_utils as hu
from haven import haven_chk as hc
from args import parse_args


def main(args):

    #------------------------ BOILERPLATE  --------------------------#

    def boilerplate(args):

        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        import uuid

        if args.model_name_impv is None:
            args.group = str(uuid.uuid1())
        else:
            args.group = args.model_name_impv + '_' + str(args.prob_statio)

        #set_seed(args, args.seed)

        return args

    args = boilerplate(args)

    def wandb_wrapper(args, first_time=True):

        if first_time:
            if not is_connected():
                print('no internet connection. Going in dry')
                os.environ['WANDB_MODE'] = 'dryrun'
            import wandb
            if args.wandb_key is not None:
                wandb.login(key=args.wandb_key)

        if args.name is None:
            wandb.init(project=args.wandb, group=args.group, reinit=True)
        else:
            wandb.init(project=args.wandb, name=args.name, group=args.group, reinit=True)
        wandb.config.update(args)

        return wandb


    #--------------------------- DATASETS ---------------------------#

    meta_train_dataloader, meta_val_dataloader, cl_dataloader = init_dataloaders(args)

    #------------------------- MODEL --------------------------------#

    def init_models(args, metalearner=None):
        if not metalearner is None:
            model = metalearner.model
        else:
            if args.pretrain_model is None:
                if args.dataset == 'omniglot':
                    model = ModelConvOmniglot(args.num_ways, hidden_size=args.hidden_size, deeper=args.deeper)
                    loss_function = F.cross_entropy
                if args.dataset == 'tiered-imagenet':
                    model = ModelConvMiniImagenet(args.num_ways, hidden_size=args.hidden_size, deeper=args.deeper)
                    loss_function = F.cross_entropy
                if args.dataset == 'synbols':
                    model = ModelConvSynbols(args.num_ways, hidden_size=args.hidden_size, deeper=args.deeper)
                    loss_function = F.cross_entropy
                if args.dataset == "harmonics":
                    #NOTE: doesn't work yet
                    model = ModelMLPSinusoid(hidden_sizes=[40, 40])
                    loss_function = F.mse_loss
            else:
                model.load_state_dict(torch.load(args.pretrain_model))

        if args.bgd_optimizer:
            meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
            meta_optimizer_cl = create_BGD_optimizer(model.to(args.device),
                                                     mean_eta=args.mean_eta,
                                                     std_init=args.std_init,
                                                     mc_iters=args.train_mc_iters)
        else:
            meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
            meta_optimizer_cl = meta_optimizer

        if metalearner is None:
            if args.method == 'MAML':
                metalearner = ModelAgnosticMetaLearning(model, meta_optimizer, loss_function, args)
            elif args.method == 'ModularMAML':
                metalearner = ModularMAML(model, meta_optimizer, loss_function, args, wandb=None)

        return metalearner, meta_optimizer, meta_optimizer_cl

    metalearner, meta_optimizer, meta_optimizer_cl = init_models(args)

    #---------------------- PRETRAINING TIME ------------------------#

    def pretraining(args, metalearner, meta_optimizer, meta_train_dataloader, meta_val_dataloader):

        if args.pretrain_model is None:

            # best_metalearner = copy.deepcopy(metalearner)
            best_metalearner = metalearner

            if args.num_epochs==0:
                best_val = None
                pass

            else:
                best_val = 0.
                epochs_overfitting = 0

                epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
                for epoch in range(args.num_epochs):

                    metalearner.train(meta_train_dataloader, max_batches=args.num_batches,
                                      verbose=args.verbose, desc='Training', leave=False)
                    results = metalearner.evaluate(meta_val_dataloader,
                                                   max_batches=args.num_batches,
                                                   verbose=args.verbose,
                                                   epoch=epoch,
                                                   desc=epoch_desc.format(epoch + 1))

                    result_val = results['accuracies_after']

                    # early stopping:
                    if (best_val is None) or (best_val < result_val):
                        epochs_overfitting = 0
                        best_val = result_val
                        best_metalearner = copy.deepcopy(metalearner)
                        if args.output_folder is not None:
                            with open(args.model_path, 'wb') as f:
                                torch.save(model.state_dict(), f)
                    else:
                        epochs_overfitting +=1
                        if epochs_overfitting > args.patience:
                            break

                print('\npretraining done!\n')
                # if wandb is not None:
                #     wandb.log({'best_val':best_val}, step=epoch)

        else:

            best_metalearner = copy.deepcopy(metalearner)

        cl_model_init = copy.deepcopy(best_metalearner)
        cl_model_init.best_pretrain_val = best_val

        del metalearner, best_metalearner

        return cl_model_init

    cl_model_init = pretraining(args, metalearner, meta_optimizer, meta_train_dataloader, meta_val_dataloader)

    #-------------------------- CL TIME -----------------------------#

    def continual_learning(args, cl_model_init, meta_optimizer_cl, cl_dataloader):

        # new args
        cl_model_init.optimizer_cl = meta_optimizer_cl
        cl_model_init.cl_strategy = args.cl_strategy
        cl_model_init.cl_strategy_thres = args.cl_strategy_thres
        cl_model_init.cl_tbd_thres = args.cl_tbd_thres

        if args.no_cl_meta_learning:
            cl_model_init.no_meta_learning = True

        mode_list = ['pretrain', 'ood1', 'ood2']

        is_classification_task = args.is_classification_task

        # keep a per run logger:
        # TODO:
        final_results = dict(zip(mode_list, [[], [], []]))
        final_results['total'] = []
        final_results['precision'], final_results['recall'], final_results['f1_score'] = [], [], []

        for run in range(args.n_runs):

            #set_seed(args, rgs.seed) if run==0 else set_seed(args, random.randint(0,100000))

            wandb = wandb_wrapper(args)

            scores = []
            precisions = []
            recalls = []
            modes= []
            scores_mode = dict(zip(mode_list, [[], [], []]))

            ## init model
            cl_model = copy.deepcopy(cl_model_init)
            _, _, meta_optimizer_cl = init_models(args, cl_model)
            cl_model.optimizer_cl = meta_optimizer_cl

            for i, batch in enumerate(cl_dataloader):

                data, labels, task_switch, mode = batch
                if args.cl_accumulate:
                    curr_results = cl_model.observe_accumulate(batch)
                else:
                    curr_results = cl_model.observe(batch)

                ## Reporting:
                mode = mode[0]

                if is_classification_task:
                    score = curr_results["accuracy_after"]
                else:
                    #TODO: redo this
                    score = results["mse_after"]

                tbd = float(curr_results['tbd'])

                scores.append(score)
                modes.append(mode)
                scores_mode[mode].append(score)


                wandb.log({"online_task_switch":task_switch}, step=i)
                tbd_score = float(task_switch == tbd)
                if task_switch:
                    precisions.append(tbd_score)
                    wandb.log({"online_precision":tbd_score}, step=i)
                else:
                    recalls.append(tbd_score)
                    wandb.log({"online_recall":tbd_score}, step=i)

                if args.verbose and (i%10==0):

                    if is_classification_task:
                        acc_so_far = np.mean(scores)
                    else:
                        mse_so_far = np.mean(scores)
                    precision_so_far = np.mean(precisions)
                    recall_so_far = np.mean(recalls)

                    message = []

                    print(
                        f"run: {run}",
                        (
                            f"total Acc: {acc_so_far:.2f}"
                            if is_classification_task else
                            f"mean MSE: {mse_so_far:.5f}"
                        ),
                        f"total Precision: {precision_so_far:.2f}",
                        f"total Recall: {recall_so_far:.2f}",
                        f"it: {i}", sep="\t"
                    )

                wandb.log({"online_acc_total":score}, step=i)
                wandb.log({"online_acc_{}".format(mode):score}, step=i)

                ## for smoothness sake, logging avg accuracy over 100 steps
                if i % 100 == 0 and i>0:
                    wandb.log({"online_acc_total_by_100":np.mean(scores[-100:])}, step=i)
                    for mode in mode_list:
                        wandb.log({"online_acc_{}_by_100".format(mode):
                                np.mean(np.array(scores[-100:])[np.array(modes[-100:])==mode])}, step=i)

                ## run finished
                if i==args.timesteps-1:

                    if wandb is not None:
                        run_acc = np.mean(scores)
                        wandb.log({'final_acc_total':run_acc}, step=i)
                        final_results['total'].append(run_acc)
                        for mode in mode_list:
                            run_acc = np.mean(np.array(scores)[np.array(modes)==mode])
                            wandb.log({'final_acc_{}'.format(mode):run_acc}, step=i)
                            final_results[mode].append(run_acc)

                        ## log TBD
                        run_precision = np.mean(precisions)
                        run_recall = np.mean(recalls)
                        wandb.log({'final_precision':run_precision}, step=i)
                        wandb.log({'final_recall':run_recall}, step=i)
                        final_results['precision'].append(run_precision)
                        final_results['recall'].append(run_recall)
                        run_f1_score = 2 * (run_precision*run_recall)/(run_precision+run_recall)
                        wandb.log({'final_f1_score':run_f1_score}, step=i)
                        final_results['f1_score'] = run_f1_score


                        #keep it open to log final results
                        if run != args.n_runs-1:
                            wandb.join()

                    ## Only when searching
                    # if run==0 and is_classification_task:
                    #     if run_acc < 1./ float(args.num_ways) + 0.1:
                    #         ## didnt beat random...
                    #         wandb.log({'fail':1})
                    #         return
                    break

        print('\n ----- Finished all runs -----\n')

        ## final reporting
        if wandb is not None:

            for key in final_results:
                avg = np.mean(final_results[key])
                std = np.std(final_results[key])
                print(f'{key}: \t {avg:.2f} +/- {std:.2f}')
                wandb.log({"final_{}_avg".format(key):avg})
                wandb.log({"final_{}_std".format(key):std})


            wandb.log({'best_pretrain_val':cl_model_init.best_pretrain_val})

    # launch CL jobs
    continual_learning(args, cl_model_init, meta_optimizer_cl, cl_dataloader)


if __name__ == "__main__":
    from args import parse_args
    args = parse_args()
    main(args)
