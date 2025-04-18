""" 
SegViT: Semantic Segmentation with Plain Vision Transformers
https://github.com/zbwxp/SegVit.git
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
import matplotlib.pyplot as plt
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class TPN_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, 
                      memory: Tensor, 
                      tgt_mask: Optional[Tensor] = None,
                      memory_mask: Optional[Tensor] = None, 
                      tgt_key_padding_mask: Optional[Tensor] = None,
                      memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn

class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size()
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ATMHead(BaseDecodeHead):
    def __init__(
            self,
            img_size_x,
            img_size_z,
            in_channels,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=3,
            use_proj=True,
            shrink_ratio=None,
            threshold=0.3, 
            **kwargs,
    ):
        super(ATMHead, self).__init__(in_channels=in_channels, **kwargs)

        self.image_size_x = img_size_x
        self.image_size_z = img_size_z
        self.use_stages = use_stages
        nhead = num_heads
        dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []
        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_(proj.weight, std=.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)
            # decoder layer
            decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
            decoder = TPN_Decoder(decoder_layer, num_layers)
            self.add_module("decoder_{}".format(i + 1), decoder)
            atm_decoders.append(decoder)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = atm_decoders
        self.q = nn.Embedding(self.num_classes, dim)

        self.class_embed = nn.Linear(dim, self.num_classes + 1)

        delattr(self, 'conv_seg')
        delattr(self, 'loss_decode')
        

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, inputs, num_token_decoder):
        x = []
        for stage_ in inputs[:self.use_stages]:
            x.append(stage_[:,-num_token_decoder:,])
        x.reverse()  
        bs = x[0].size()[0]

        attns_x = []
        qs = []
        out = {}
        q = self.q.weight.repeat(bs, 1, 1).transpose(0, 1)  

        for idx, (x_, proj_, norm_, decoder_) in enumerate(zip(x, self.input_proj, self.proj_norm, self.decoder)):
            lateral = norm_(proj_(x_))
            q, attn = decoder_(q, lateral.transpose(0, 1))
            
            qs.append(q.transpose(0, 1))
            
            attn = attn.transpose(-1, -2)
            attn_x = self.d3_to_d4(attn[:, -num_token_decoder:]) 
            attns_x.append(attn_x)
            
        qs = torch.stack(qs, dim=0)  
        outputs_class = self.class_embed(qs)[-1] 
        out["pred_logits"] = outputs_class
        
        seg_masks_x = torch.sum(torch.stack(attns_x, dim=0), dim=0) 
        
        masks_x = self.semantic_inference(outputs_class, seg_masks_x)
        
        masks_x_full = F.interpolate(masks_x, size=(self.image_size_x, self.image_size_x), mode='bilinear', align_corners=False)
        masks_x_full_flatten = masks_x_full.flatten(1)
        out["masks_x"] = masks_x
        out["pred_masks_full"] = masks_x_full_flatten
        
        return out["masks_x"], out["pred_masks_full"]


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]  
        mask_pred = mask_pred.sigmoid()   
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)  
        return semseg

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)


def build_decoder(cfg, hidden_dim=768, pretrained=False):
    if cfg.MODEL.DECODER.ADD_DECODER:
        decoder = ATMHead(
                    img_size_x=cfg.DATA.SEARCH.SIZE,
                    img_size_z=cfg.DATA.TEMPLATE.SIZE,
                    in_channels=hidden_dim,
                    channels=hidden_dim,
                    num_classes=1,
                    num_layers=cfg.MODEL.DECODER.NUM_LAYERS,
                    num_heads=12,
                    use_stages=len(cfg.MODEL.BACKBONE.OUT_INDICES), 
                    embed_dims=cfg.MODEL.DECODER.DECODER_DIM,
                    )
        
        def _adjust_and_load_weights(pretrained_model):
            decoder_dict = {}
            skip_keys = [
                'decode_head.loss_decode.criterion.empty_weight',
                'decode_head.q.weight',
                'decode_head.class_embed.weight',
                'decode_head.class_embed.bias',
                'decode_head.mask_embed.layers.0.weight',
                'decode_head.mask_embed.layers.0.bias',
                'decode_head.mask_embed.layers.1.weight',
                'decode_head.mask_embed.layers.1.bias',
                'decode_head.mask_embed.layers.2.weight',
                'decode_head.mask_embed.layers.2.bias',
            ]
            for old_name, param in pretrained_model.items():
                # Skip specific keys
                if old_name in skip_keys:
                    continue
                # Adjust key name by removing 'decode_head.'
                if 'decode_head' in old_name:
                    new_name = old_name.replace('decode_head.', '')
                    decoder_dict[new_name] = param
            return decoder_dict
    
        if pretrained:
            checkpoint = torch.load(pretrained, map_location="cpu")
            decoder_dict = _adjust_and_load_weights(checkpoint["state_dict"])
            missing_keys, unexpected_keys = decoder.load_state_dict(decoder_dict, strict=False)
            print('Load pretrained decoder from: ' + pretrained)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)


    else:
        decoder = nn.Identity()
        print('decoder is not used in the application.')

    return decoder