import torch
import torch.nn.functional as F

import math
import os
import time
import json
import random
import logging
import numpy as np
import copy
from pdb import set_trace

from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Compose
from dataloaders import init_dataloaders

from MAML.model import ModelConvSynbols, ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
from MAML.metalearners import ModelAgnosticMetaLearning, ModularMAML, DynamicModularMAML
from MAML.utils import ToTensor1D, set_seed, is_connected

from Utils.bgd_lib.bgd_optimizer import create_BGD_optimizer


def main(args):

    #------------------------ BOILERPLATE  --------------------------#

    def boilerplate(args):

        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #set_seed(args, args.seed)

        if args.wandb is not None:
            if not is_connected():
                print('no internet connection. Going in dry')
                os.environ['WANDB_MODE'] = 'dryrun'
            import wandb

            if args.wandb_key is not None:
                wandb.login(key=args.wandb_key)
            if args.name is None:
                wandb.init(project=args.wandb)
            else:
                wandb.init(project=args.wandb, name=args.name)
            wandb.config.update(args)
        else:
            wandb=None

        return args, wandb

    args, wandb = boilerplate(args)

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
                metalearner = ModularMAML(model, meta_optimizer, loss_function, args, wandb=wandb)

        return metalearner, meta_optimizer, meta_optimizer_cl

    metalearner, meta_optimizer, meta_optimizer_cl = init_models(args)

    #---------------------- PRETRAINING TIME ------------------------#

    def pretraining(args, metalearner, meta_optimizer, meta_train_dataloader, meta_val_dataloader):

        if args.pretrain_model is None:

            # best_metalearner = copy.deepcopy(metalearner)
            best_metalearner = metalearner

            if args.num_epochs==0:
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
                if wandb is not None:
                    wandb.log({'best_val':best_val}, step=epoch)

        else:

            best_metalearner = copy.deepcopy(metalearner)

        cl_model_init = copy.deepcopy(best_metalearner)

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

        modes = ['train', 'test', 'ood']

        is_classification_task = args.is_classification_task

        accuracies = np.zeros([args.n_runs, args.timesteps])
        mses = np.zeros([args.n_runs, args.timesteps])

        tbds = np.zeros([args.n_runs, args.timesteps])
        avg_accuracies_mode = dict(zip(modes, [[], [], []]))
        avg_mses_mode = dict(zip(modes, [[], [], []]))



        for run in range(args.n_runs):

            #set_seed(args, rgs.seed) if run==0 else set_seed(args, random.randint(0,100000))

            accuracies_mode = dict(zip(modes, [[], [], []]))
            mses_mode = dict(zip(modes, [[], [], []]))

            cl_model = copy.deepcopy(cl_model_init)
            _, _, meta_optimizer_cl = init_models(args, cl_model)
            cl_model.optimizer_cl = meta_optimizer_cl

            for i, batch in enumerate(cl_dataloader):

                data, labels, task_switch, mode = batch
                results = cl_model.observe(batch)

                ## Reporting:
                if is_classification_task:
                    accuracy_after = results["accuracy_after"]
                    accuracies[run, i] = accuracy_after
                    accuracies_mode[mode[0]].append(accuracy_after)
                else:
                    mse_after = results["mse_after"]
                    mses[run, i] = mse_after
                    mses_mode[mode[0]].append(mse_after)

                tbds[run, i] = float(results['tbd'])

                if wandb is not None and run==0:
                    if is_classification_task:
                        accuracy_after = results["accuracy_after"]
                        wandb.log({'temp_cl_acc': accuracy_after, 'timestep1':i})
                        wandb.log({
                            f'temp_cl_acc_{mode[0]}': accuracy_after,
                            'timestep2':i
                        })
                    else:
                        mse_after = results["mse_after"]
                        wandb.log({
                            'temp_cl_mse': mse_after,
                            'timestep1': i,
                        })
                        wandb.log({
                            f'temp_cl_mse_{mode[0]}': mse_after,
                            'timestep2': i
                        })

                if args.verbose or i==args.timesteps-1:
                    if is_classification_task:
                        acc = np.mean(accuracies[run,:i])
                    else:
                        mse = np.mean(mses[run, :i])
                    tbd = np.mean(tbds[run,:i])

                    message = []

                    print(
                        (
                            f"total Acc: {acc:.2f}"
                            if is_classification_task else
                            f"mean MSE: {mse:.5f}"
                            f" MSE: {mses[run, i]:.3f}"
                        ),

                        f"Total tbd: {tbd:.2f}",
                        f"it: {i}", sep="\t"
                    )

                if i==args.timesteps-1:
                    for mode in modes:
                        avg_accuracies_mode[mode].append(np.mean(accuracies_mode[mode]))

                    if wandb is not None:
                        wandb.log({'cl_acc_by_runs':np.mean(accuracies[run,:]), 'run':run})

                    if run==0 and is_classification_task:
                        if acc < 1./ float(args.num_ways) + 0.2:
                            wandb.log({'fail':1})
                            return
                    break


        if wandb is not None:
            # avg accuracy per time steps:
            for i in range(args.timesteps):

                wandb.log({'cl_acc':np.mean(accuracies[:,i]), 'timestep3':i})
                wandb.log({'cl_acc_std':np.std(accuracies[:,i]), 'timestep4':i})

            # final avgs:
            final_accs = np.mean(accuracies, axis=1)
            wandb.log({'final_acc':np.mean(final_accs)})
            wandb.log({'final_acc_std':np.std(final_accs)})

            final_begin_accs = np.mean(accuracies[:,:args.timesteps], axis=1)
            wandb.log({'final_begin_acc':np.mean(final_begin_accs)})
            wandb.log({'final_begin_acc_std':np.std(final_begin_accs)})

            final_tbds = np.mean(tbds, axis=1)
            wandb.log({'final_tbd':np.mean(final_tbds)})
            wandb.log({'final_tbd_std':np.std(final_tbds)})

            for mode in modes:
                wandb.log({'final_{}'.format(mode):np.mean(avg_accuracies_mode[mode])})
                wandb.log({'final_{}_std'.format(mode):np.std(avg_accuracies_mode[mode])})

    continual_learning(args, cl_model_init, meta_optimizer_cl, cl_dataloader)


if __name__ == "__main__":
    from args import parse_args
    args = parse_args()
    main(args)


