import torch
from torch.optim.lr_scheduler import _LRScheduler

class LinearDecayWithWarmup(_LRScheduler):
    def __init__(self, optimizer, total_iters, increase_iters, base_lr=1e-8, max_lr=5e-4, last_epoch=-1):
        self.total_iters = total_iters
        self.increase_iters = increase_iters
        self.base_lr = base_lr
        self.max_lr = max_lr
        super(LinearDecayWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_iter = self.last_epoch + 1
        if current_iter <= self.increase_iters:
            # Linear increase
            lr = self.base_lr + (self.max_lr - self.base_lr) * (current_iter / self.increase_iters)
        else:
            # Linear decrease
            lr = self.max_lr - (self.max_lr - self.base_lr) * ((current_iter - self.increase_iters) / (self.total_iters - self.increase_iters))
        return [lr for _ in self.optimizer.param_groups]