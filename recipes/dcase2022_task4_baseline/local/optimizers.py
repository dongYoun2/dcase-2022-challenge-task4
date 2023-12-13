from torch import optim


def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, **kwargs):
    return optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
