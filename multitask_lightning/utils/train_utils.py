from functools import partial

from torch import optim, nn

def get_optim(optim_config):
    optimizers = {'adam': optim.Adam,
                  'sgd': optim.SGD,
                  'adamw': optim.AdamW,
                  'adadelta': optim.Adadelta,
                  }

    try:
        optimizer = optimizers[optim_config.pop('name')]
    except KeyError:
        raise ValueError('Optimizer {} is not recognized'.format(optim_config['name']))
    return partial(optimizer, **optim_config)

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep