from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_xywh
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap_floor, generate_heatmap
from lib.train.data.grid_generator import unwarp_bboxes_batch, warp_boxes
import torch.nn.functional as F


class FocusTrackActor(BaseActor):
    """ Actor for training FocusTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.use_grid = cfg.DATA.SEARCH.GRID.USE_GRID

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        if self.cfg.MODEL.DECODER.ADD_DECODER:
            assert 'full_mask' in pred_dict, "Decoder outputs are missing!"
        
        loss = 0
        status = {}
        b = gt_dict['is_negative_pair'].shape[0]
        is_negative_pair = gt_dict['is_negative_pair']
        positive_mask = (is_negative_pair==0).long()    
        
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            print(pred_boxes)
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        
        if self.use_grid:
            pred_boxes_vec = unwarp_bboxes_batch(box_cxcywh_to_xyxy(pred_dict['pred_boxes']), gt_dict['search_grids'][0]).view(-1, 4)
            warped_box_xyxy = warp_boxes(box_xywh_to_xyxy(gt_dict['search_anno'].permute(1,0,2)), gt_dict['search_grids'][0])
            warped_box = box_xyxy_to_xywh(warped_box_xyxy)
            gt_gaussian_maps = generate_heatmap_floor(warped_box.permute(1,0,2), (self.cfg.DATA.SEARCH.SIZE, self.cfg.DATA.SEARCH.SIZE), self.cfg.MODEL.BACKBONE.STRIDE)
            gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
            gt_gaussian_maps = gt_gaussian_maps * positive_mask.view(-1, 1, 1, 1)
        else:
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_dict['pred_boxes']).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
            gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
            gt_gaussian_maps = gt_gaussian_maps * positive_mask.view(-1, 1, 1, 1)
            
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        if self.loss_weight['giou'] > 0:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            giou_loss = (giou_loss * positive_mask).sum() / positive_mask.sum()
            mean_iou = (iou.detach() * positive_mask).sum() / positive_mask.sum()
            loss += giou_loss * self.loss_weight['giou'] 
            status["Loss/giou"] = giou_loss.item()
        
        # compute l1 loss
        if self.loss_weight['l1'] > 0:
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec, reduction='none')  
            l1_loss = (l1_loss * positive_mask.unsqueeze(1)).sum() / positive_mask.sum()
            loss += self.loss_weight['l1'] * l1_loss
            status["Loss/l1"] = l1_loss.item()  
        
        # compute location loss
        if 'score_map' in pred_dict and self.loss_weight['focal'] > 0:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
            loss += self.loss_weight['focal'] * location_loss
            status["Loss/location"] = location_loss.item()
   
        
        if self.cfg.MODEL.DECODER.ADD_DECODER:
            
            full_mask_pred = pred_dict['full_mask'] 
            search_masks = gt_dict['search_masks'].squeeze().flatten(1)
            full_mask_gt = search_masks
            full_mask_gt = full_mask_gt * positive_mask[:, None] 
            
            if self.loss_weight['focal_mask'] > 0:
                focal_mask_loss = self.objective['focal_mask'](full_mask_pred, full_mask_gt)
                loss += self.loss_weight['focal_mask'] * focal_mask_loss
                status["Loss/focal_mask"] = focal_mask_loss.item()
                
        if self.cfg.TRAIN.TRAIN_SECOND_STAGE and self.loss_weight['logits'] > 0:
            assert 'logits' in pred_dict
            logits = pred_dict['logits'].reshape(-1,2)
            logits_gt = positive_mask.reshape(-1) 
            logits_loss = self.objective['logits'](logits, logits_gt) 
            loss += self.loss_weight['logits'] * logits_loss
            status["Loss/logits"] = logits_loss.item()
        
        if return_status:
            status["Loss/total"] = loss.item()
            status["IoU"] = mean_iou.item()
            return loss, status
        else:
            return loss