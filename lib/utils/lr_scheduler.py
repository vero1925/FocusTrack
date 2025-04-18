
# import math

# import torch


# class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
#     """cosine annealing scheduler with warmup.

#     Args:
#         optimizer (Optimizer): Wrapped optimizer.
#         T_max (int): Maximum number of iterations.
#         eta_min (float): Minimum learning rate. Default: 0.
#         last_epoch (int): The index of last epoch. Default: -1.

#     .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
#         https://arxiv.org/abs/1608.03983
#     """

#     def __init__(
#         self,
#         optimizer,
#         T_max,
#         eta_min,
#         warmup_factor=1.0 / 3,
#         warmup_iters=500,
#         warmup_method="linear",
#         last_epoch=-1,
#     ):
#         if warmup_method not in ("constant", "linear"):
#             raise ValueError(
#                 "Only 'constant' or 'linear' warmup_method accepted"
#                 "got {}".format(warmup_method)
#             )

#         self.T_max = T_max
#         self.eta_min = eta_min
#         self.warmup_factor = warmup_factor
#         self.warmup_iters = warmup_iters
#         self.warmup_method = warmup_method
#         super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         if self.last_epoch < self.warmup_iters:
#             return self.get_lr_warmup()
#         else:
#             return self.get_lr_cos_annealing()

#     def get_lr_warmup(self):
#         if self.warmup_method == "constant":
#             warmup_factor = self.warmup_factor
#         elif self.warmup_method == "linear":
#             alpha = self.last_epoch / self.warmup_iters
#             warmup_factor = self.warmup_factor * (1 - alpha) + alpha
#         return [
#             base_lr * warmup_factor
#             for base_lr in self.base_lrs
#         ]

#     def get_lr_cos_annealing(self):
#         last_epoch = self.last_epoch - self.warmup_iters
#         T_max = self.T_max - self.warmup_iters
#         return [self.eta_min + (base_lr - self.eta_min) *
#                 (1 + math.cos(math.pi * last_epoch / T_max)) / 2
#                 for base_lr in self.base_lrs]



# class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
#     """multi-step learning rate scheduler with warmup."""

#     def __init__(
#         self,
#         optimizer,
#         milestones,
#         gamma=0.1,
#         warmup_factor=1.0 / 3,
#         warmup_iters=500,
#         warmup_method="linear",
#         last_epoch=-1,
#     ):
#         if not list(milestones) == sorted(milestones):
#             raise ValueError(
#                 "Milestones should be main.tex list of" " increasing integers. Got {}",
#                 milestones,
#             )

#         if warmup_method not in ("constant", "linear"):
#             raise ValueError(
#                 "Only 'constant' or 'linear' warmup_method accepted"
#                 "got {}".format(warmup_method)
#             )
#         self.milestones = milestones
#         self.gamma = gamma
#         self.warmup_factor = warmup_factor
#         self.warmup_iters = warmup_iters
#         self.warmup_method = warmup_method
#         super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         warmup_factor = 1
#         if self.last_epoch < self.warmup_iters:
#             if self.warmup_method == "constant":
#                 warmup_factor = self.warmup_factor
#             elif self.warmup_method == "linear":
#                 alpha = self.last_epoch / self.warmup_iters
#                 warmup_factor = self.warmup_factor * (1 - alpha) + alpha
#          return[
#             base_lr
#             * warmup_factor
#             * self.gamma ** bisect_right(self.milestones, self.last_epoch)
#             for base_lr in self.base_lrs
#         ]
