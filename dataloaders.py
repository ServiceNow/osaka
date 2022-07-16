import sys
import torch
import numpy as np
from pdb import set_trace
import os

# --------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------

def select_from_tensor(tensor, index):
    """ equivalent to tensor[index] but for batched / 2D+ tensors """

    last_dim = index.dim() - 1

    assert tensor.dim() >= index.dim()
    assert index.size()[:last_dim] == tensor.size()[:last_dim]

    # we have to make `train_idx` the same shape as train_data, or else
    # `torch.gather` complains.
    # see https://discuss.pytorch.org/t/batched-index-select/9115/5

    missing_dims = tensor.dim() - index.dim()
    index = index.view(index.size() + missing_dims * (1,))
    index = index.expand((-1,) * (index.dim() - missing_dims) + tensor.size()[(last_dim+1):])

    return torch.gather(tensor, last_dim, index)

def order_and_split(data_x, data_y):
    """ given a dataset, returns (num_classes, samples_per_class, *data_x[0].size())
        tensor where samples (and labels) are ordered and split per class """

    xx, yy = [], []
    for label in data_y.unique():
        idx = torch.where(data_y == label)[0]
        xx += [data_x[idx]]
        yy += [data_y[idx]]

    data_x, data_y = xx, yy

    # give equal amt of points for every class
    #TODO(if this is restrictive for some dataset, we can change)
    min_amt  = min([x.size(0) for x in data_x])
    data_x   = torch.stack([x[:min_amt] for x in data_x])
    data_y   = torch.stack([y[:min_amt] for y in data_y])

    # sanity check
    for i, item in enumerate(data_y):
        assert item.unique().size(0) == 1 and item[0] == i, 'wrong result'

    return data_x, data_y


# --------------------------------------------------------------------------
# Datasets and Streams (the good stuff)
# --------------------------------------------------------------------------

class MetaDataset(torch.utils.data.Dataset):
    """ Dataset similar to BatchMetaDataset in TorchMeta """

    def __init__(self, train_data, test_data,  n_shots_tr, n_shots_te, n_way,
                    args=None, **kwargs):

        '''
        Parameters
        ----------

        train_data : Array of (x,) pairs, one for each class. Contains all the
            training data that should be available at meta-training time (inner loop).
        test_data  : Array of (x,) pairs, one for each class. These are the
            same classes as in `train_data`. Used at meta-testing time (outer loop).
        n_way      : number of classes per task at meta-testing
        n_shots_tr : number of samples per classes
        n_shots_te : number of samples per classes

        '''

        # NOTE: for now assume train_data and test_data have shape
        # (n_classes, n_samples_per_task, *data_shape).

        # TODO: should torchvision transforms be passed in here ?

        # separate the classes into tasks
        n_classes   = len(train_data)

        self._len        = None
        self.n_way       = n_way
        self.kwargs      = kwargs
        self.n_classes   = n_classes
        self.n_shots_tr  = n_shots_tr
        self.n_shots_te  = n_shots_te

        if args is None:
            self.input_size  = [28,28]
            self.device      = 'cpu'
            self.is_classification_task = True
        else:
            self.input_size  = args.input_size
            self.device      = args.device
            self.is_classification_task = args.is_classification_task

        self.all_classes = np.arange(n_classes)

        self.train_data  = train_data
        self.test_data   = test_data

        if args.dataset == 'tiered-imagenet':
            self.cpu_dset = True
        else:
            self.cpu_dset = False

    def __len__(self):
        # return the number of train / test batches that can be built
        # without sample repetition
        if self._len is None:
            n_samples = sum([x.shape[0] for x in self.train_data])
            self._len = n_samples // (self.n_way * (self.n_shots_tr + self.n_shots_te))

        return self._len


    def __getitem__(self, index):
        if self.is_classification_task:
            return self._getitem_classification(index)
        else:
            return self._getitem_regression(index)

    def _getitem_regression(self, index):
        train_x = self.train_data[..., 0, None]
        train_y = self.train_data[..., 1, None]
        test_x = self.test_data[..., 0, None]
        test_y = self.test_data[..., 1, None]

        if self.cpu_dset:
            train_x = train_x.to(self.device)
            train_y = train_y.to(self.device)
            test_x = test_x.to(self.device)
            test_y = test_y.to(self.device)

        return {
            "train": [train_x, train_y],
            "test": [test_x, test_y],
        }

    def _getitem_classification(self, index):
        # NOTE: This method COMPLETELY ignores the index. This will be a problem
        # if you wish to recover a specific batch of data.

        classes_in_task = np.random.choice(self.all_classes, self.n_way, replace=False)
        train_samples_in_class = self.train_data.shape[1]
        test_samples_in_class  = self.test_data.shape[1]

        train_data = self.train_data[classes_in_task]
        test_data  = self.test_data[classes_in_task]

        # sample indices for meta train
        train_idx = torch.Tensor(self.n_way, self.n_shots_tr)
        if not(self.cpu_dset):
            train_idx = train_idx.to(self.device)
        train_idx = train_idx.uniform_(0, train_samples_in_class).long()

        # samples indices for meta test
        test_idx = torch.Tensor(self.n_way, self.n_shots_te)
        if not(self.cpu_dset):
            test_idx = test_idx.to(self.device)
        test_idx = test_idx.uniform_(0, test_samples_in_class).long()

        train_x = select_from_tensor(train_data, train_idx)
        test_x  = select_from_tensor(test_data,  test_idx)

        train_x = train_x.view(-1, *self.input_size)
        test_x = test_x.view(-1, *self.input_size)

        # build label tensors
        train_y = torch.arange(self.n_way).view(-1, 1).expand(-1, self.n_shots_tr)
        train_y = train_y.flatten()

        test_y  = torch.arange(self.n_way).view(-1, 1).expand(-1, self.n_shots_te)
        test_y  = test_y.flatten()

        if self.cpu_dset:
            train_x = train_x.float().to(self.device)
            train_y = train_y.to(self.device)
            test_x = test_x.float().to(self.device)
            test_y = test_y.to(self.device)

        #return train_x, train_y, test_x, test_y

        # same signature are TorchMeta
        out = {}
        out['train'], out['test'] = [train_x,train_y], [test_x, test_y]

        return out

class StreamDataset(torch.utils.data.Dataset):
    """ stream of non stationary dataset as described by Mass """

    def __init__(self, train_data, test_data, ood_data, n_shots=1,
            n_way=5, prob_statio=.8, prob_train=0.1, prob_test=0.8,
            prob_ood=0.1, args=None, **kwargs):

        '''
        Parameters
        ----------

        train_data : Array of (x,) pairs, one for each class. Contains the SAME
            classes used during (meta) training, but different samples.
        test_data  : Array of (x,) pairs, one for each class. These are DIFFERENT
            classes from the ones used during (meta) training.
        n_way      : number of classes per task at cl-test time
        n_shots    : number of samples per classes at cl-test time

        '''

        assert prob_train + prob_test + prob_ood == 1.
        if args.dataset == 'tiered-imagenet':
            self.cpu_dset = True
        else:
            self.cpu_dset = False

        self.n_shots    = n_shots
        self.n_way      = n_way

        self.modes    = ['train', 'test', 'ood']
        self.modes_id = [0, 1, 2]
        self.probs    = np.array([prob_train, prob_test, prob_ood])
        self.data     = [train_data, test_data, ood_data]
        self.p_statio = prob_statio

        self.task_sequence: List[str] = []
        self.n_steps_per_task = 1
        self.index_in_task_sequence = 0
        self.steps_done_on_task = 0

        if args is None:
            self.input_size  = [28,28]
            self.device      = 'cpu'
            self.is_classification_task = True
        else:
            self.input_size  = args.input_size
            self.device      = args.device
            self.is_classification_task = args.is_classification_task
            self.task_sequence = args.task_sequence
            self.n_steps_per_task = args.n_steps_per_task

        self.mode_name_map = dict(zip(self.modes, self.modes_id))

        # mode in which to start ( 0 --> 'train' )
        self._mode = 0
        self._classes_in_task = None
        self._samples_in_class = None


    def __len__(self):
        # this is a never ending stream
        return sys.maxsize


    def __getitem__(self, index):
        if self.is_classification_task:
            return self._getitem_classification(index)
        else:
            return self._getitem_regression(index)

    def _getitem_regression(self, index):
        task_switch = False
        if self.task_sequence:
            self.steps_done_on_task += 1

            if self.steps_done_on_task >= self.n_steps_per_task:
                task_switch = True
                self.steps_done_on_task = 0
                self.index_in_task_sequence += 1
                self.index_in_task_sequence %= len(self.task_sequence)

            mode_name = self.task_sequence[self.index_in_task_sequence]
            self._mode = self.mode_name_map[mode_name]
        else:
            if (np.random.uniform() > self.p_statio):
                mode  = np.random.choice(self.modes_id, p=self.probs)
                self._mode = mode
                task_switch = mode != self._mode

        mode_data = self.data[self._mode]

        x = mode_data[..., 0, None]
        y = mode_data[..., 1, None]
        if self.cpu_dset:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y, task_switch, self.modes[self._mode]

    def _getitem_classification(self, index):
        # NOTE: This method COMPLETELY ignores the index. This will be a problem
        # if you wish to recover a specific batch of data.

        # NOTE: using multiple workers (`num_workers > 0`) or `batch_size  > 1`
        # will have undefined behaviour. This is because unlike regular datasets
        # here the sampling process is sequential.
        task_switch = 0
        if self.task_sequence:
            self.steps_done_on_task += 1

            if self.steps_done_on_task >= self.n_steps_per_task:
                task_switch = 1
                self.steps_done_on_task = 0
                self.index_in_task_sequence += 1
                self.index_in_task_sequence %= len(self.task_sequence)

            mode_name = self.task_sequence[self.index_in_task_sequence]
            self._mode = self.mode_name_map[mode_name]
        elif (np.random.uniform() > self.p_statio) or (self._classes_in_task is None):
            # mode  = np.random.choice(self.modes_id, p=self.probs)
            # self._mode = mode
            # task_switch = mode != self._mode
            # TODO: this makes a switch even if staying in same mode!
            task_switch = 1
            self._mode  = np.random.choice([0,1,2], p=self.probs)

            mode_data = self.data[self._mode]
            n_classes = len(mode_data)
            self._samples_in_class = mode_data.size(1)

            # sample `n_way` classes
            self._classes_in_task = np.random.choice(np.arange(n_classes), self.n_way,
                    replace=False)

        else:

            task_switch = 0

        mode_data = self.data[self._mode]
        data = mode_data[self._classes_in_task]

        # sample indices for meta train
        idx = torch.Tensor(self.n_way, self.n_shots)#.to(self.device)
        idx = idx.uniform_(0, self._samples_in_class).long()
        if not(self.cpu_dset):
            idx = idx.to(self.device)
        data = select_from_tensor(data, idx)

        # build label tensors
        labels = torch.arange(self.n_way).view(-1, 1).expand(-1, self.n_shots).to(self.device)

        # squeeze
        data = data.view(-1, *self.input_size)
        labels = labels.flatten()

        if self.cpu_dset:
            data = data.float().to(self.device)
            labels = labels.to(self.device)

        return data, labels, task_switch, self.modes[self._mode]



def init_dataloaders(args):

    if args.dataset == 'omniglot':
        from Data.omniglot import Omniglot
        from torchvision.datasets import MNIST, FashionMNIST

        args.is_classification_task = True
        args.prob_train, args.prob_test, args.prob_ood = 0.5, 0.25, 0.25
        args.n_train_cls = 900
        args.n_val_cls = 100
        args.n_train_samples = 10


        args.input_size = [1,28,28]
        Omniglot_dataset = Omniglot(args.folder).data
        Omniglot_dataset = torch.from_numpy(Omniglot_dataset).type(torch.float).to(args.device)
        meta_train_dataset = Omniglot_dataset[:args.n_train_cls]
        meta_train_train = meta_train_dataset[:,:args.n_train_samples,:,:]
        meta_train_test = meta_train_dataset[:,args.n_train_samples:,:,:]

        meta_val_dataset = Omniglot_dataset[args.n_train_cls : (args.n_train_cls+args.n_val_cls)]
        #TODO(figure out the bug when there is only a single class fed to the valid dataloader)
        meta_val_train = meta_val_dataset[:,:args.n_train_samples,:,:]
        meta_val_test = meta_val_dataset[:,args.n_train_samples:,:,:]

        cl_dataset = Omniglot_dataset
        cl_ood_dataset1 = MNIST(args.folder, train=True,  download=True)
        cl_ood_dataset2 = FashionMNIST(args.folder, train=True,  download=True)
        cl_ood_dataset1, _ = order_and_split(cl_ood_dataset1.data, cl_ood_dataset1.targets)
        cl_ood_dataset2, _ = order_and_split(cl_ood_dataset2.data, cl_ood_dataset2.targets)
        cl_ood_dataset1 = cl_ood_dataset1[:,:,None,:,:]
        cl_ood_dataset2 = cl_ood_dataset2[:,:,None,:,:]
        cl_ood_dataset1 = cl_ood_dataset1.type(torch.float).to(args.device)
        cl_ood_dataset2 = cl_ood_dataset2.type(torch.float).to(args.device)


    elif args.dataset == "tiered-imagenet":
        from Data.tiered_imagenet import NonEpisodicTieredImagenet

        args.prob_train, args.prob_test, args.prob_ood = 0.3, 0.3, 0.4

        args.is_classification_task = True
        args.n_train_cls = 100
        args.n_val_cls = 100
        args.n_train_samples = 500

        args.input_size = [3,64,64]
        tiered_dataset = NonEpisodicTieredImagenet(args.folder, split="train")

        meta_train_dataset = tiered_dataset.data[:args.n_train_cls]
        meta_train_train = meta_train_dataset[:,:args.n_train_samples, ...]
        meta_train_test = meta_train_dataset[:,args.n_train_samples:,...]

        meta_val_dataset = tiered_dataset.data[args.n_train_cls : (args.n_train_cls+args.n_val_cls)]
        meta_val_train = meta_val_dataset[:,:args.n_train_samples,:,:]
        meta_val_test = meta_val_dataset[:,args.n_train_samples:,:,:]

        cl_dataset = tiered_dataset.data
        set_trace()

        cl_ood_dataset1 = tiered_dataset.data[(args.n_train_cls+args.n_val_cls):]
        ## last results computed with this split
        #cl_ood_dataset1 = tiered_dataset.data[200:300]
        cl_ood_dataset2 = NonEpisodicTieredImagenet(args.folder, split="val").data
        #cl_dataset = cl_dataset.type(torch.float)#.to(args.device)
        cl_ood_dataset1 = cl_ood_dataset1.type(torch.float)#.to(args.device)
        cl_ood_dataset2 = cl_ood_dataset2.type(torch.float)#.to(args.device)


    elif args.dataset == "harmonics":
        '''under construction'''
        from  data.harmonics import Harmonics
        args.is_classification_task = False
        args.input_size = [1]

        def make_dataset(train: bool = True) -> torch.Tensor:
            return torch.from_numpy(
                Harmonics(train=train).data
            ).float()

        dataset = make_dataset()
        meta_train_dataset = dataset[:500]
        meta_train_train = meta_train_dataset[:, :40]
        meta_train_test  = meta_train_dataset[:, 40:]

        meta_val_dataset = dataset[500:]
        meta_val_train = meta_val_dataset[:, :40]
        meta_val_test  = meta_val_dataset[:, 40:]

        if args.mode=='train':
            cl_dataset = dataset
            cl_ood_dataset1 = make_dataset(train=False)
            cl_ood_dataset2 = make_dataset(train=False)
            cl_ood_dataset3 = make_dataset(train=False)


        args.prob_train, args.prob_test, args.prob_ood = 0.6, 0., 0.4

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(args.dataset))

    meta_train_dataloader = MetaDataset(meta_train_train, meta_train_test, args=args,
            n_shots_tr=args.num_shots, n_shots_te=args.num_shots_test, n_way=args.num_ways)
    meta_val_dataloader = MetaDataset(meta_val_train, meta_val_test, args=args,
            n_shots_tr=args.num_shots, n_shots_te=args.num_shots_test, n_way=args.num_ways)

    meta_train_dataloader = torch.utils.data.DataLoader(meta_train_dataloader,
            batch_size=args.batch_size)
    meta_val_dataloader = torch.utils.data.DataLoader(meta_val_dataloader,
            batch_size=args.batch_size)

    cl_dataloader = StreamDataset(cl_dataset, cl_ood_dataset1, cl_ood_dataset2,
            n_shots=args.num_shots, n_way=args.num_ways, prob_statio=args.prob_statio,
            prob_train=args.prob_train, prob_test=args.prob_test, prob_ood=args.prob_ood, args=args)
    cl_dataloader = torch.utils.data.DataLoader(cl_dataloader, batch_size=1)

    del meta_train_dataset, meta_train_train, meta_train_test, meta_val_dataset,\
            meta_val_train, meta_val_test, cl_dataset, cl_ood_dataset1, cl_ood_dataset2

    return meta_train_dataloader, meta_val_dataloader, cl_dataloader



