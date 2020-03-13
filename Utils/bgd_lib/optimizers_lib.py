import torch.optim as optim
from .bgd_optimizer import BGD


def bgd(model, **kwargs):
    #logger = kwargs.get("logger", None)
    #assert(logger is not None)
    bgd_params = {
        "mean_eta": kwargs.get("mean_eta", 1),
        "std_init": kwargs.get("std_init", 0.02),
        "mc_iters": kwargs.get("mc_iters", 10)
    }
    #logger.info("BGD params: " + str(bgd_params))
    all_params = [{'params': params} for l, (name, params) in enumerate(model.named_parameters())]
    return BGD(all_params, **bgd_params)