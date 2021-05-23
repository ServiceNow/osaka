import torch
from pdb import set_trace
import numpy as np
import random

from collections import OrderedDict
from torchmeta.modules import MetaModule

class BinaryLayer(torch.autograd.Function):
    def __init__(self):
        super(BinaryLayer, self).__init__()
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return torch.sign(input)
    @staticmethod
    def backward(self, grad_output):
        #input = self.saved_tensors[0]
        #grad_output[input>1]=0
        #grad_output[input<-1]=0
        return grad_output

class ReluTroughLayer(torch.autograd.Function):
    def __init__(self):
        super(ReluTroughLayer, self).__init__()
    @staticmethod
    def forward(self, input):
        return torch.relu(input)
    @staticmethod
    def backward(self, grad_output):
        return grad_output

def update_parameters(model, loss, params=None, step_size=0.5, first_order=False,
            freeze_visual_features=False, no_meta_learning=False, step_size_activation=None):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.

    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.

    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.

    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.
    """
    if not isinstance(model, MetaModule):
        raise ValueError()

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss, params.values(),
        create_graph=not first_order, allow_unused=True)

    out = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        #NOTE: this is for meta-SGD loop
        for (name, param), grad in zip(params.items(), grads):
            if (freeze_visual_features and 'visual_features' in name) or no_meta_learning:
                out[name] = param
            else:
                if step_size_activation is None:
                    out[name] = param - step_size[name] * grad
                elif step_size_activation == 'binary_trough':
                    out[name] = param - 0.5*(BinaryLayer.apply(step_size[name])+1) * grad
                elif step_size_activation == 'relu_trough':
                    out[name] = param - ReluTroughLayer.apply(step_size[name]) * grad
    else:
        for (name, param), grad in zip(params.items(), grads):
            if (freeze_visual_features and 'visual_features' in name) or no_meta_learning:
                out[name] = param
            else:
                out[name] = param - step_size * grad

    return out

def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()

def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
            for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
            for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()

def set_seed(args, manualSeed):
    #####seed#####
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    if args.device != "cpu":
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    ######################################################

def is_connected(host='http://google.com'):
    import urllib.request
    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except:
        return False


class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.

    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'
