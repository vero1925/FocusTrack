import torch
from torch.utils.data.distributed import DistributedSampler

import lib.train.data.transforms as tfm
# from bisect import bisect_right

from lib.train.data import sampler, opencv_loader, processing, LTRLoader
from lib.train.base_functions import names2datasets
from lib.models.focustrack import build_focustrack
from lib.train.actors import FocusTrackActor
from lib.utils.focal_loss import FocalLoss, sigmoid_focal_loss
from lib.utils.box_ops import giou_loss, SIoU_loss, ciou
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from lib.utils.misc import is_main_process
from lib.train.data.grid_generator import build_data_search_grid_generator
from lib.train.trainers import LTRTrainer
# from lib.utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR


def build_processing(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    # ToTensorAndJitter: brightness jitter
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor
 
    output_sz['search'] = cfg.DATA.SEARCH.HEIGHT
    grid_generator = build_data_search_grid_generator(cfg)
    settings.center_jitter_factor['correct'] = cfg.DATA.SEARCH.RANDOM_CENTER_JITTER
    settings.scale_jitter_factor['correct'] = cfg.DATA.SEARCH.RANDOM_SCALE_JITTER
    use_grid = {'search':cfg.DATA.SEARCH.GRID.USE_GRID, 'template':cfg.DATA.TEMPLATE.USE_GRID}
    data_processing_train = processing.QpProcessing(search_area_factor=search_area_factor,
                                                    output_sz=output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    transform=transform_train,
                                                    joint_transform=transform_joint,
                                                    settings=settings,
                                                    grid_generator=grid_generator,
                                                    jitter_anno_prob=cfg.DATA.SEARCH.RANDOM_JITTER_RATIO,
                                                    use_grid=use_grid,
                                                    gt_gauss_generator=cfg.DATA.SEARCH.GT_GAUSS_TYPE,
                                                    multi_jitter=cfg.DATA.SEARCH.MULTI_JITTER)
    return data_processing_train

def build_dataloaders(cfg, settings):
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))
    
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    output_sz['search'] = cfg.DATA.SEARCH.SIZE
    grid_generator = build_data_search_grid_generator(cfg)
    settings.center_jitter_factor['correct'] = cfg.DATA.SEARCH.RANDOM_CENTER_JITTER
    settings.scale_jitter_factor['correct'] = cfg.DATA.SEARCH.RANDOM_SCALE_JITTER
    use_grid = {'search':cfg.DATA.SEARCH.GRID.USE_GRID, 'template':cfg.DATA.TEMPLATE.USE_GRID}
    pos_prob = getattr(cfg.DATA.TRAIN, "POS_PROB", 1.0)
    print("pos_prob", pos_prob)
    
    data_processing_train = processing.QpProcessing(search_area_factor=search_area_factor,
                                                    output_sz=output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    transform=transform_train,
                                                    joint_transform=transform_joint,
                                                    settings=settings,
                                                    grid_generator=grid_generator,
                                                    jitter_anno_prob=cfg.DATA.SEARCH.RANDOM_JITTER_RATIO,
                                                    use_grid=use_grid)
    
    data_processing_val = processing.QpProcessing(search_area_factor=search_area_factor,
                                                    output_sz=output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    transform=transform_val,
                                                    joint_transform=transform_joint,
                                                    settings=settings,
                                                    grid_generator=grid_generator,
                                                    jitter_anno_prob=cfg.DATA.SEARCH.RANDOM_JITTER_RATIO,
                                                    use_grid=use_grid)  
    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls, 
                                            pos_prob=pos_prob,
                                            seed=settings.local_rank)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_cls , seed=settings.local_rank)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val

def build_net(cfg, settings):
    net = build_focustrack(cfg, settings=settings)
    return net

def build_actor(cfg, settings, net):
    assert cfg.TRAIN.CE_WEIGHT is not None, "CE_WEIGHT must be set when using negative sample"
    focal_loss = FocalLoss()
    objective = {'giou': giou_loss, 
                'l1': l1_loss, 
                'focal': focal_loss, 
                'logits': CrossEntropyLoss(),
                'focal_mask':sigmoid_focal_loss,
                }
    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 
                    'l1': cfg.TRAIN.L1_WEIGHT, 
                    'focal': cfg.TRAIN.FOCALLOSS_WEIGHT, 
                    'logits': cfg.TRAIN.CE_WEIGHT,
                    'focal_mask':cfg.TRAIN.FOCALMASK_WEIGHT,
                    }
    
    print("loss_weight", loss_weight)
    actor = FocusTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    return actor

def build_optimizer_scheduler(cfg, actor):
    train_second_stage = cfg.TRAIN.TRAIN_SECOND_STAGE   
    
    if train_second_stage and 'FocusTrack' in cfg.MODEL.PRETRAIN_FILE :
        # only update cls head
        param_dicts = [
            {
                "params": [p for n, p in actor.net.named_parameters() if "cls_head" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR,
            },]
        for n, p in actor.net.named_parameters():
            if "cls_head" not in n:
                p.requires_grad = False
                
        for n, p in actor.net.named_parameters():
            if p.requires_grad:
                print(n)
    else:
        param_dicts = [
            {"params": [p for n, p in actor.net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in actor.net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in actor.net.named_parameters():
                if p.requires_grad:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    
    # scheduler
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    elif cfg.TRAIN.SCHEDULER.TYPE == "CosineAnnealingLR":
        eta_min=0
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.EPOCH, eta_min=eta_min, last_epoch=-1)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler

def build_trainer(actor, loader, optimizer, settings, lr_scheduler, cfg):
    save_epochs = get_save_epochs(cfg)
    trainer = LTRTrainer(actor, loader, optimizer, settings, lr_scheduler, save_epochs=save_epochs)
    return trainer


def get_save_epochs(cfg):
    epoch_list = [i for i in range(cfg.TRAIN.SAVE_INTERVAL, cfg.TRAIN.EPOCH, cfg.TRAIN.SAVE_INTERVAL)]
    epoch_list2 = cfg.TRAIN.EXTRA_SAVE_EPOTH
    
    for ep in epoch_list2:
        if ep in epoch_list:
            continue
        else:
            epoch_list.append(ep)

    if cfg.TRAIN.EPOCH > 99:
        threshold = int(cfg.TRAIN.EPOCH / 2)
        epoch_list = [ep for ep in epoch_list if ep > threshold]
    
    return epoch_list