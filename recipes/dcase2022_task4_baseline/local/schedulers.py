import numpy as np
from desed_task.utils import schedulers


def ExponentialWarmup(optimizer, max_lr, steps_per_epoch, n_epochs_warmup, exponent=-5.0, **kwargs):
    """Scheduler to apply ramp-up during training to the learning rate.
    Args:
        optimizer: torch.optimizer.Optimizer, the optimizer from which to rampup the value from
        max_lr: float, the maximum learning to use at the end of ramp-up.
        rampup_length: int, the length of the rampup (number of steps).
        exponent: float, the exponent to be used.
    """
    return schedulers.ExponentialWarmup(optimizer, max_lr, steps_per_epoch * n_epochs_warmup, exponent)


class ExponentialWarmupDecay(schedulers.ExponentialWarmup):
    def __init__(self, optimizer, max_lr, steps_per_epoch, n_epochs_warmup, n_epochs, exponent=-5.0, **kwargs):
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs_warmup = n_epochs_warmup
        self.n_epochs = n_epochs
        self.rampup_len = self.n_epochs_warmup * self.steps_per_epoch
        self.rampdown_len = (self.n_epochs - self.n_epochs_warmup) * self.steps_per_epoch
        super().__init__(optimizer, max_lr, self.rampup_len, exponent)

    def _get_lr(self):
        if self.rampup_len == 0:
            scale = 1.0
        else:
            if self.step_num <= self.rampup_len:
                scale = self._get_scaling_factor()
            else:
                phase = (self.step_num - self.rampup_len) / self.rampdown_len
                scale = float(np.exp(self.exponent * phase * phase))
        return self.max_lr * scale
