import torch
import random
import numpy as np

def log_string(log, string):
    """
    print log information
    :param log: file pointer
    :param string: string to write to file
    :return:
    """
    log.write(string + '\n')
    log.flush()
    print(string)

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_device(opt):
    if torch.cuda.is_available():
        opt.cuda = True
        torch.cuda.set_device(int(opt.device[5]))
    else:
        opt.cuda = False
        opt.device = 'cpu'
    return opt

def init_optim(model, opt):
    # Initialize the optimizer
    return torch.optim.Adam(params=model.parameters(),lr=opt.lr_init)

def init_lr_scheduler(optim, opt):
    # Initialize the learning strategy
    # return torch.optim.lr_scheduler.StepLR(optimizer=optim,gamma=opt.lr_scheduler_rate,step_size=opt.lr_scheduler_step)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=opt.lr_decay_steps,
                                                gamma = opt.lr_scheduler_rate)

def print_model_parameters(model, only_num = True):
    # Record the trainable parameters of the model
    if not only_num:
        for name, param in model.named_parameters():
            continue
    total_num = sum([param.nelement() for param in model.parameters()])
    print('模型可训练参数: {:,}'.format(total_num))

def get_memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024*1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024*1024.)
    # print('Allocated Memory: {:.2f} MB, Cached Memory: {:.2f} MB'.format(allocated_memory, cached_memory))
    return allocated_memory, cached_memory
    