import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool2d(nn.Module):
    def __init__(self, embed_dim: int, num_tokens_all: int, num_heads: int = 12, output_dim: int = None):
        super().__init__()
        self.num_cls, self.num_token_z, self.num_token_x = num_tokens_all
        self.positional_embedding = nn.Parameter(torch.randn(self.num_cls+self.num_token_x, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.initialize_parameters()

    def initialize_parameters(self):
        std = self.c_proj.in_features ** -0.5
        nn.init.normal_(self.q_proj.weight, std=std)
        nn.init.normal_(self.k_proj.weight, std=std)
        nn.init.normal_(self.v_proj.weight, std=std)
        nn.init.normal_(self.c_proj.weight, std=std)
    
    def forward(self, x):  
        cls_token = x[:, 0:1, :]  # B 1 C
        search_token = x[:, -self.num_token_x:, :]  # B N C
        x = torch.cat([cls_token, search_token], dim=1)  # B N+1 C
        
        x = x.permute(1, 0, 2)  # NLC -> LNC 
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        
        x_out, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x_out.squeeze(0)


class LogitsPredictor(nn.Module,):
    def __init__(self, embed_dim=768, num_tokens_all=321, num_classes=2, average_pool= False, average_pool_type='avg_pool', head_type='linear'):
        super(LogitsPredictor, self).__init__()
        
        self.num_classes = num_classes
        self.average_pool = average_pool
        if self.average_pool:
            self.average_pool_type = average_pool_type
            if self.average_pool_type == 'avg_pool': 
                self.avgpool = nn.AdaptiveAvgPool1d(1)
            elif self.average_pool_type == 'attention_pool':
                self.avgpool = AttentionPool2d(embed_dim = embed_dim, num_tokens_all=num_tokens_all,)
  
        self.head_type = head_type
        if self.head_type == 'linear':
            self.mlp_head = nn.Linear(embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        elif self.head_type == 'mlp':
            self.mlp_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, self.num_classes)) if self.num_classes > 0 else nn.Identity()
        else:
            raise ValueError("MLP HEAD TYPE %s is not supported." % self.head_type)
            
    def forward(self, x):
        if self.average_pool:
            if self.average_pool_type == 'avg_pool': 
                logits = self.avgpool(x.transpose(1, 2))  # B C 1
            elif self.average_pool_type == 'attention_pool':
                logits = self.avgpool(x)  # B C 1            
            logits = torch.flatten(logits, 1)   # B C
        else:
            logits = x[:, 0]   # B C
        logits = self.mlp_head(logits)
        
        return logits
    

def build_cls_head(cfg, hidden_dim=768, num_tokens_all=[1, 64, 256]):
    if cfg.MODEL.HEAD.CLS_HEAD.USE_CLS_TOKEN:
        cls_head = LogitsPredictor(embed_dim=hidden_dim,
                                   num_tokens_all = num_tokens_all,
                                    num_classes = cfg.MODEL.HEAD.CLS_HEAD.NUM_CLASSES, 
                                    average_pool= cfg.MODEL.HEAD.CLS_HEAD.AVERAGE_POOL, 
                                    average_pool_type = cfg.MODEL.HEAD.CLS_HEAD.AVERAGE_POOL_TYPE, 
                                    head_type = cfg.MODEL.HEAD.CLS_HEAD.HEAD_TYPE)
    else:
        cls_head = nn.Identity()
        print('Cls_head is not used in the application.')

    return cls_head
