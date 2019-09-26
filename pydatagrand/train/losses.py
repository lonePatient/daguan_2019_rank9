# encoding:utf-8
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss

__call__ = ['CrossEntropy', 'BCEWithLogLoss']


class CrossEntropy(object):
    def __init__(self, ignore_index=-1):
        self.loss_f = CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss


class BCEWithLogLoss(object):
    def __init__(self):
        self.loss_fn = BCEWithLogitsLoss()

    def __call__(self, output, target):
        loss = self.loss_fn(input=output, target=target)
        return loss


class SpanLoss(object):
    def __init__(self, ignore_index=-100):
        self.loss_fn = CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, output, target, mask):
        active_loss = mask.view(-1) == 1
        active_logits = output[active_loss]
        active_labels = target[active_loss]
        return self.loss_fn(active_logits, active_labels)
