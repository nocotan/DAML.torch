# -*- coding: utf-8 -*-
from daml.functions import pair_loss


class DMLUpdater(object):
    def __init__(self, **kwargs):
        self.optimizer = kwargs.pop("optimizer")
        self.model = kwargs.pop("model")
        self.loss_type = kwargs.pop("loss_type")

    def update(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()

        if self.loss_type == "pair":
            self.update_pair(inputs, targets)
        elif self.loss_type == "triplet":
            self.update_triplet(inputs, targets)
        else:
            raise NotImplementedError

    def update_pair(self, inputs, targets):
        outputs = self.model(*inputs)
        loss = pair_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

        return loss
