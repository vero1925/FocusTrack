"""
Basic FocusTrack model.
"""
import math
import os
from typing import List
import re
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.box_head import build_box_head
from lib.models.layers.cls_head import build_cls_head
from lib.models.layers.decoder import build_decoder
from lib.models.focustrack.vit import vit_base_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh


class FocusTrack(nn.Module):
    """ This is the base class for FocusTrack """

    def __init__(self, transformer, decoder, box_head, cls_head, aux_loss=False, 
                 box_head_type="CORNER", 
                 use_cls_head="False",):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.num_token_z = self.backbone.pos_embed_z.shape[1]
        self.num_token_x = self.backbone.pos_embed_x.shape[1]
        self.nun_clstoken = 1 if self.backbone.add_cls_token else 0
        self.decoder = decoder if decoder is not nn.Identity() else None
        self.box_head = box_head
        if use_cls_head :
            self.cls_head = cls_head
        self.aux_loss = aux_loss
        self.box_head_type = box_head_type
        if box_head_type == "CORNER" or box_head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        self.num_token_decoder = self.num_token_x
        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
    def forward(self, template: torch.Tensor, search: torch.Tensor, training=True):
        x, aux_dict = self.backbone(z=template, x=search,)

        # backbone feature
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        
        # decoder
        if 'intermediate_outputs' in aux_dict and type(self.decoder) != nn.Identity:
            intermediate_outputs = aux_dict['intermediate_outputs']
            masks_x, pred_masks_full = self.decoder(intermediate_outputs, 
                                                        self.num_token_decoder,)
        else: 
            masks_x = None
            pred_masks_full = None 
            
        out = self.forward_box_head(cat_feature=feat_last, 
                                    mask_feature=masks_x, 
                                    gt_score_map=None,)
        
        if self.decoder is not None:
            # out['feat_mask'] = masks_x 
            out['full_mask'] = pred_masks_full
        
        if 'before_norm' in aux_dict:
            logits = self.cls_head(aux_dict['before_norm'])
            out['logits'] = logits
        out.update(aux_dict)
        # out['backbone_feat'] = x  # for debugging
        return out

    def forward_box_head(self, cat_feature, mask_feature=None, gt_score_map=None,):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)  # (b, 768, 16, 16)

        if self.box_head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, mask_feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.box_head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, mask_feature, gt_score_map,)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_focustrack(cfg, settings=None, training=True):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('FocusTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, 
                                        drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        use_cls_token=cfg.MODEL.HEAD.CLS_HEAD.USE_CLS_TOKEN,
                                        num_classes=cfg.MODEL.HEAD.CLS_HEAD.NUM_CLASSES,
                                        out_indices=cfg.MODEL.BACKBONE.OUT_INDICES,)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
        backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
        num_tokens_all = backbone.num_tokens_all

    else:
        raise NotImplementedError

    box_head = build_box_head(cfg, hidden_dim, add_decoder=cfg.MODEL.DECODER.ADD_DECODER)
    cls_head = build_cls_head(cfg, hidden_dim=hidden_dim, num_tokens_all=num_tokens_all)
        
    # decoder pretrained model 
    if cfg.MODEL.DECODER.PRETRAIN_FILE  and training:
        pretrained_decoder = os.path.join(pretrained_path, cfg.MODEL.DECODER.PRETRAIN_FILE)
    else:
        pretrained_decoder = ''
    decoder = build_decoder(cfg, hidden_dim=hidden_dim,
                            pretrained=pretrained_decoder,)

    model = FocusTrack(
        backbone,
        decoder,
        box_head,
        cls_head,
        aux_loss=False,
        box_head_type=cfg.MODEL.HEAD.BOX_HEAD.TYPE,
        use_cls_head=cfg.MODEL.HEAD.CLS_HEAD.USE_CLS_TOKEN,
    )

    if 'FocusTrack' in cfg.MODEL.PRETRAIN_FILE and cfg.TRAIN.TRAIN_SECOND_STAGE and training:
        if settings is None:
            raise ValueError('settings is None')
        script_name = settings.script_name
        config_name = settings.config_name
        config_name_former = '{}_stage1'.format( re.sub(r'_[^_]+$', '', config_name) )
        pretrained_path = os.path.join(current_dir, '../../../output/checkpoints/train/', script_name, config_name_former, cfg.MODEL.PRETRAIN_FILE)
        
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
