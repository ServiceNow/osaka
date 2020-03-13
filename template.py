from dataloaders import init_dataloaders


def main(args):


    #--------------------------- BOILERPLATE ------------------------#
    args.device = 'cpu'

    #--------------------------- DATASETS ---------------------------#

    meta_train_dataloader, meta_val_dataloader, cl_dataloader = init_dataloaders(args)


    #---------------------- PRETRAINING TIME ------------------------#


    if args.pretrain_model is None:

        print('Pretraining time')
        if args.num_epochs==0:
            pass

        else:
            for epoch in range(args.num_epochs):

                for batch in meta_train_dataloader:
                    '''
                    batch = {'train', 'test'}
                    batch['train'][0] = batch-size x num_shots*num_ways x input_dim
                    batch['train'][1] = batch-size x num_shots*num_ways x output_dim
                    batch['test'][0]  = batch-size x num_shots-test*num_ways x input_dim
                    batch['test'][1]  = batch-size x num_shots-test*num_ways x output_dim
                    '''
                    pass

                for batch in meta_val_dataloader:
                    pass

    #-------------------------- CL TIME -----------------------------#


    print('Continual learning time')
    for run in range(args.n_runs):

        for i, batch in enumerate(cl_dataloader):

            data, labels, task_switch, mode = batch


if __name__ == "__main__":
    from args import parse_args
    args = parse_args()
    main(args)


