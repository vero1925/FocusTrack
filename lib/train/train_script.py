import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related

# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.focustrack import build_focustrack
# forward propagation related
from lib.train.actors import FocusTrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    builder_module = importlib.import_module("lib.train.train_settings.%s" % settings.script_name)

    # Build dataloaders
    loader = getattr(builder_module, 'build_dataloaders')(cfg, settings)

    # Create network
    net = getattr(builder_module, 'build_net')(cfg, settings)

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    
    # Loss functions and Actors
    actor = getattr(builder_module, 'build_actor')(cfg, settings, net)

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = getattr(builder_module, 'build_optimizer_scheduler')(cfg, actor)
    trainer = getattr(builder_module, 'build_trainer')(actor, loader, optimizer, settings, lr_scheduler, cfg)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
