import math
import numpy as np
import warnings
from torch.optim.optimizer import Optimizer
from ..common.tools import logger
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['CustomDecayLR',
           'BertLR',
           'CyclicLR',
           'ReduceLROnPlateau',
           'ReduceLRWDOnPlateau',
           'CosineLRWithRestarts',
           'NoamLR',
           'OneCycleScheduler'
           ]

class CustomDecayLR(object):
    '''
    自定义学习率变化机制
        Example:
        >>> scheduler = CustomDecayLR(optimizer)
        >>> for epoch in range(100):
        >>>     scheduler.epoch_step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>     validate(...)
    '''
    def __init__(self,optimizer,lr):
        self.optimizer = optimizer
        self.lr = lr

    def epoch_step(self,epoch):
        lr = self.lr
        if epoch > 12:
            lr = lr / 1000
        elif epoch > 8:
            lr = lr / 100
        elif epoch > 4:
            lr = lr / 10
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class BertLR(object):
    '''
    Bert模型内定的学习率变化机制
    Example:
        >>> scheduler = BertLR(optimizer)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    '''
    def __init__(self,optimizer,lr,t_total,warmup):
        self.lr = lr
        self.optimizer = optimizer
        self.t_total = t_total
        self.warmup = warmup

    # 线性预热方式
    def warmup_linear(self,x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def batch_step(self,training_step):
        lr_this_step = self.lr * self.warmup_linear(training_step / self.t_total,self.warmup)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_this_step

class CyclicLR(object):
    '''
    Cyclical learning rates for training neural networks
    Example:
        >>> scheduler = CyclicLR(optimizer)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    '''
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')

        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} base_lr, got {len(base_lr)}")
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} max_lr, got {len(max_lr)}")
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class ReduceLROnPlateau(object):
    """Reduce learning rate when a metrics has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.


    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_acc, val_loss = validate(...)
        >>>     scheduler.epoch_step(val_loss, epoch)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=0, epsilon=1e-4, cooldown=0, min_lr=0,eps=1e-8):

        super(ReduceLROnPlateau, self).__init__()
        assert isinstance(optimizer, Optimizer)
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience - 1
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.monitor_op = None
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.optimizer = optimizer
        self.eps = eps
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['min', 'max']:
            raise RuntimeError('Learning Rate Plateau Reducing mode %s is unknown!')
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def reset(self):
        self._reset()

    def epoch_step(self, metrics, epoch):
        current = metrics
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires metrics available!', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0

            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        old_lr = float(param_group['lr'])
                        if old_lr > self.min_lr + self.eps:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            param_group['lr'] = new_lr
                            if self.verbose > 0:
                                logger.info(f'\nEpoch {epoch}: reducing learning rate to {new_lr}.')
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                self.wait += 1


    def in_cooldown(self):
        return self.cooldown_counter > 0

class ReduceLRWDOnPlateau(ReduceLROnPlateau):
    """Reduce learning rate and weight decay when a metrics has stopped
    improving. Models often benefit from reducing the learning rate by
    a factor of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate and weight decay factor is reduced for
    optimizers that implement the the weight decay method from the paper
    `Fixing Weight Decay Regularization in Adam`_.

    .. _Fixing Weight Decay Regularization in Adam:
        https://arxiv.org/abs/1711.05101
    for AdamW or SGDW
    Example:
        >>> optimizer = AdamW(model.parameters(), lr=0.1, weight_decay=1e-3)
        >>> scheduler = ReduceLRWDOnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.epoch_step(val_loss)
    """
    def epoch_step(self, metrics, epoch):
        current = metrics
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires metrics available!', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        old_lr = float(param_group['lr'])
                        if old_lr > self.min_lr + self.eps:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            param_group['lr'] = new_lr
                            if self.verbose > 0:
                                logger.info(f'Epoch {epoch}: reducing learning rate to {new_lr}.')
                        if param_group['weight_decay'] != 0:
                            old_weight_decay = float(param_group['weight_decay'])
                            new_weight_decay = max(old_weight_decay * self.factor, self.min_lr)
                            if old_weight_decay > new_weight_decay + self.eps:
                                param_group['weight_decay'] = new_weight_decay
                                if self.verbose:
                                    logger.info(f'\nEpoch {epoch}: reducing weight decay factor of group to {new_weight_decay:.4e}.')
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
                self.wait += 1

class CosineLRWithRestarts(object):
    """Decays learning rate with cosine annealing, normalizes weight decay
    hyperparameter value, implements restarts.
    https://arxiv.org/abs/1711.05101

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        batch_size: minibatch size
        epoch_size: training samples per epoch
        restart_period: epoch count in the first restart period
        t_mult: multiplication factor by which the next restart period will extend/shrink

    Example:
        >>> scheduler = CosineLRWithRestarts(optimizer, 32, 1024, restart_period=5, t_mult=1.2)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    """

    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,
                 t_mult=2, last_epoch=-1, eta_threshold=1000, verbose=False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   f"in param_groups[{i}] when resuming an"
                                   " optimizer")
        self.base_lrs = list(map(lambda group: group['initial_lr'],
                                 optimizer.param_groups))

        self.last_epoch = last_epoch
        self.batch_size = batch_size
        self.iteration = 0
        self.epoch_size = epoch_size
        self.eta_threshold = eta_threshold
        self.t_mult = t_mult
        self.verbose = verbose
        self.base_weight_decays = list(map(lambda group: group['weight_decay'],
                                           optimizer.param_groups))
        self.restart_period = restart_period
        self.restarts = 0
        self.t_epoch = -1
        self.batch_increments = []
        self._set_batch_increment()

    def _schedule_eta(self):
        """
        Threshold value could be adjusted to shrink eta_min and eta_max values.
        """
        eta_min = 0
        eta_max = 1
        if self.restarts <= self.eta_threshold:
            return eta_min, eta_max
        else:
            d = self.restarts - self.eta_threshold
            k = d * 0.09
            return (eta_min + k, eta_max - k)

    def get_lr(self, t_cur):
        eta_min, eta_max = self._schedule_eta()

        eta_t = (eta_min + 0.5 * (eta_max - eta_min)
                 * (1. + math.cos(math.pi *
                                  (t_cur / self.restart_period))))

        weight_decay_norm_multi = math.sqrt(self.batch_size /
                                            (self.epoch_size *
                                             self.restart_period))
        lrs = [base_lr * eta_t for base_lr in self.base_lrs]
        weight_decays = [base_weight_decay * eta_t * weight_decay_norm_multi
                         for base_weight_decay in self.base_weight_decays]

        if self.t_epoch % self.restart_period < self.t_epoch:
            if self.verbose:
                logger.info(f"Restart at epoch {self.last_epoch}")
            self.restart_period *= self.t_mult
            self.restarts += 1
            self.t_epoch = 0

        return zip(lrs, weight_decays)

    def _set_batch_increment(self):
        d, r = divmod(self.epoch_size, self.batch_size)
        batches_in_epoch = d + 2 if r > 0 else d + 1
        self.iteration = 0
        self.batch_increments = list(np.linspace(0, 1, batches_in_epoch))

    def batch_step(self):
        self.last_epoch += 1
        self.t_epoch += 1
        self._set_batch_increment()
        try:
            t_cur = self.t_epoch + self.batch_increments[self.iteration]
            self.iteration += 1
        except (IndexError):
            raise RuntimeError("Epoch size and batch size used in the "
                               "training loop and while initializing "
                               "scheduler should be the same.")

        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups,self.get_lr(t_cur)):
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay

class NoamLR(object):
    '''
    主要参考论文<< Attention Is All You Need>>中的学习更新方式
    Example:
        >>> scheduler = NoamLR(d_model,factor,warm_up,optimizer)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         glopab_step += 1
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step(global_step)
        >>>     validate(...)
    '''
    def __init__(self,d_model,factor,warm_up,optimizer):
        self.optimizer = optimizer
        self.warm_up = warm_up
        self.factor = factor
        self.d_model = d_model
        self._lr = 0

    def get_lr(self,step):
        lr = self.factor * (self.d_model ** (-0.5) * min(step ** (-0.5),step * self.warm_up ** (-1.5)))
        return lr

    def batch_step(self,step):
        '''
        update parameters and rate
        :return:
        '''
        lr = self.get_lr(step)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self._lr = lr

class OneCycleScheduler(_LRScheduler):
    """Implements the One Cycle scheduler from https://arxiv.org/pdf/1803.09820.pdf
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_size (int): Number of training iterations to be performed
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        warmup_ratio (float): ratio of iterations used to reach max_lr
        phases (tuple): specify the scaling mode of both phases (possible values: 'linear', 'cosine')
        base_ratio (float): ratio between base_lr and max_lr during warmup phase
        final_ratio (float): ratio between base_lr and max_lr during last phase
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
    """

    def __init__(self,
                 optimizer,
                 total_size,
                 max_lr=None,
                 warmup_ratio=0.3,
                 phases=None,
                 base_ratio=0.2,
                 final_ratio=None,
                 cycle_momentum=True,
                 base_momentum=0.8,
                 max_momentum=0.9,
                 last_epoch=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Specify max lr
        if isinstance(max_lr, float):
            self.max_lrs = [max_lr for group in optimizer.param_groups]
        elif isinstance(max_lr, (list, tuple)):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} values for max_lr, got {len(max_lr)}")
            self.max_lrs = max_lr
        else:
            # Take current value as max_lr
            self.max_lrs = [group['lr'] for group in optimizer.param_groups]

        # Take the division factor for each phase
        self.base_ratio = base_ratio
        self.final_ratio = base_ratio * 1e-4 if final_ratio is None else final_ratio

        self.total_size = total_size
        self.warmup_ratio = warmup_ratio

        # Phases
        self.phases = phases if isinstance(phases, tuple) else ('linear', 'cosine')
        modes = ['linear', 'cosine']
        if any(phase not in modes for phase in self.phases):
            raise ValueError(f"Phases can only take values from {modes}")

        # Handle momentum for specific optimizer
        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if 'momentum' not in optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')

            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for momentum, group in zip(base_momentums, optimizer.param_groups):
                    group['momentum'] = momentum
            self.base_momentums = list(map(lambda group: group['momentum'], optimizer.param_groups))
            self.max_momentums = self._format_param('max_momentum', optimizer, max_momentum)

        super(OneCycleScheduler, self).__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
        step_ratio = self.last_epoch / self.total_size
        # Get phase progress and LR divider for current phase
        if step_ratio <= self.warmup_ratio:
            phase_idx = 0
            x = step_ratio / self.warmup_ratio
            base_ratio = self.base_ratio
        else:
            phase_idx = 1
            x = (step_ratio - self.warmup_ratio) / (1 - self.warmup_ratio)
            base_ratio = self.final_ratio

        # Adapt scaling based on phase mode
        if self.phases[phase_idx] == 'linear':
            scale_factor = x
        elif self.phases[phase_idx] == 'cosine':
            scale_factor = 0.5 * (1 + math.cos(x * math.pi))

        # Populate LR for each group
        lrs = []
        for max_lr in self.max_lrs:
            base_lr = base_ratio * max_lr
            base_height = (max_lr - base_lr) * scale_factor
            lrs.append(base_lr + base_height)

        # Populate momentum for each group
        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_momentum = base_ratio * max_momentum
                base_height = (max_momentum - base_momentum) * scale_factor
                momentums.append(max_momentum - base_height)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

        return lrs

    def __repr__(self):
        return f"{self.__class__.__name__}(max_lr={max(self.max_lrs)}, warmup_ratio={self.warmup_ratio}, base_ratio={self.base_ratio}, final_ratio={self.final_ratio}, phases={self.phases})"

#
class BERTReduceLROnPlateau(object):
    def __init__(self, optimizer, lr,mode='min', factor=0.1, patience=10,
                 verbose=0, epsilon=1e-8, cooldown=0, min_lr=0,eps=1e-8):

        super(BERTReduceLROnPlateau, self).__init__()
        assert isinstance(optimizer, Optimizer)
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience - 1
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.monitor_op = None
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.optimizer = optimizer
        self.eps = eps
        self.lr = lr
        self.update = 0
        self._reset()

    def _reset(self):
        if self.mode not in ['min', 'max']:
            raise RuntimeError('Learning Rate Plateau Reducing mode %s is unknown!')
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def reset(self):
        self._reset()

    def epoch_step(self, metrics, epoch):
        current = metrics
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires metrics available!', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0
            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    if self.update == 0:
                        self.lr = self.lr * 1
                    else:
                        self.lr = self.lr * self.factor
                    self.update += 1
                    for param_group in self.optimizer.param_groups:
                        if self.lr > self.min_lr + self.eps:
                            new_lr = max(self.lr, self.min_lr)
                            param_group['lr'] = new_lr
                            param_group['warmup'] = 0.0
                            if self.verbose > 0:
                                logger.info(f'\nEpoch {epoch}: reducing learning rate to {new_lr}.')
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                self.wait += 1
    def in_cooldown(self):
        return self.cooldown_counter > 0