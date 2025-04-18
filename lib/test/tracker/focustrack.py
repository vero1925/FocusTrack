import math

from lib.models.focustrack import build_focustrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# for debug
import cv2
import os
import numpy as np

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.train.data.grid_generator import unwarp_bboxes, unwarp_bboxes_batch, QPGrid
from lib.utils.box_ops import box_cxcywh_to_xywh
from lib.test.tracker.vis_utils import vis_attn_maps, vis_feature_maps, vis_attn_distance


class FocusTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(FocusTrack, self).__init__(params)
        network = build_focustrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.feat_sz = self.cfg.TEST.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), 
                                     centered=True).cuda()
        
        self.use_grid = self.cfg.TEST.SEARCH.USE_GRID
        if self.use_grid:
            loss_dict = dict()
            for loss_name, loss_weight in zip(self.cfg.TEST.SEARCH.GRID.GENERATOR.LOSS.NAMES, self.cfg.TEST.SEARCH.GRID.GENERATOR.LOSS.WEIGHTS):
                loss_dict[loss_name] = loss_weight
            self.search_grid_generator = QPGrid(
                amplitude_scale=1,
                bandwidth_scale=self.cfg.TEST.SEARCH.GRID.GENERATOR.BANDWIDTH_SCALE,
                grid_type='qp',
                zoom_factor=self.cfg.TEST.SEARCH.GRID.GENERATOR.ZOOM_FACTOR,
                loss_dict=loss_dict,
                grid_shape=self.params.cfg.TEST.SEARCH.GRID.SHAPE
            )
            self.out_shape = (self.params.cfg.TEST.SEARCH.SIZE, self.params.cfg.TEST.SEARCH.SIZE)

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.base_factor = self.params.search_factor
        self.search_factor = self.params.search_factor
        self.flag = 1
        self.use_hann=True

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None

        # save states
        self.init_box = info['init_bbox']
        self.state = info['init_bbox']
        self.former_box = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}
                
        # 计算 max_search_region
        self.W, self.H = image.shape[1], image.shape[0]
        
        self.max_search_area_factor = self.cfg.TEST.MAX_SEARCH_FACTOR
        self.step = self.cfg.TEST.ENLARGE_STEP
        self.T_logits = self.cfg.TEST.T_LOGITS
        self.T_score = self.cfg.TEST.T_SCORE
        
        
    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        W_H = np.array([W, H])
        self.frame_id += 1
        self.former_box = self.state
        
        if not self.use_grid:
            x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
        else:
            x_patch_arr, x_amask_arr, resize_factor = sample_target(image, self.state, self.search_factor)  # (x1, y1, w, h)
            
            img_shape_ori = x_patch_arr.shape
            img_metas = [
                {'pad_shape': img_shape_ori}
            ]
            img_tensor = torch.tensor(x_patch_arr, dtype=torch.float32).permute(2,0,1)[None,...]
            x_patch_h, x_patch_w, _ = img_shape_ori
            saliency_box = torch.tensor(self.state, dtype=torch.float32)
            saliency_box[0] = x_patch_w / 2 - saliency_box[2] / 2
            saliency_box[1] = x_patch_h / 2 - saliency_box[3] / 2

            grid, saliency = self.search_grid_generator.forward(img_tensor.cuda(), img_metas, saliency_box[None,None,:].cuda(), self.out_shape, jitter=0, mode='center')
            x_patch_arr = F.grid_sample(img_tensor.cuda(), grid, align_corners=True)
            x_patch_arr = x_patch_arr.cpu()[0].permute(1,2,0).numpy()    # (256, 256, 3)
        
        
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)   # 经过reshape的search
    
        with torch.no_grad():
            x_dict = search
            out_dict = self.network.forward(template=self.z_dict1.tensors, search=x_dict.tensors, training=False)
        
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = out_dict['score_map']
        if self.use_hann:
            response = response * self.output_window

        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        if not self.use_grid:
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        else:
            pred_box = pred_boxes.mean(dim=0, keepdim=True)  # (cxywh)
            search_whwh = torch.tensor([self.params.search_size, self.params.search_size, self.params.search_size, self.params.search_size], device=pred_box.device)

            pred_box = torch.cat([pred_box[:,:2]-pred_box[:,2:]/2, pred_box[:,:2]+pred_box[:,2:]/2], dim=-1)   # (xywh)
            unwraped_box = unwarp_bboxes(pred_box * search_whwh, grid[0], img_shape_ori)
            unwraped_box = torch.cat([(unwraped_box[:,2:]+unwraped_box[:,:2])/2, unwraped_box[:,2:]-unwraped_box[:,:2]], dim=-1).cpu()
            if self.debug:
                # only display top-k box
                feat_x = self.cfg.TEST.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
                feat_y = self.cfg.TEST.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
                feat_xyxy = torch.tensor([feat_x, feat_y, feat_x, feat_y])
                topk_score, topk_index = torch.topk(pred_score_map.squeeze().flatten(), 4)
                topk_index2d = torch.stack([topk_index // pred_score_map.shape[-1], topk_index % pred_score_map.shape[-1]], dim=1)
                topk_pred_boxes_raw = self.cal_box_from_coord(out_dict, topk_index2d.to(out_dict['score_map'].device), resize_factor, H , W, reduce='raw')
                topk_pred_boxes_raw_xyxy = torch.cat([topk_pred_boxes_raw[:,:2]-topk_pred_boxes_raw[:,2:]/2, topk_pred_boxes_raw[:,:2]+topk_pred_boxes_raw[:,2:]/2], dim=-1)
                unwraped_topk_pred_boxes_raw = unwarp_bboxes(topk_pred_boxes_raw_xyxy * search_whwh.cuda(), grid[0], img_shape_ori)
                full_grid = F.interpolate(grid.permute(0,3,1,2), scale_factor=resize_factor)

                # wrong gt_box to make program work
                gt_box = transform_image_to_crop(torch.tensor(info['gt_bbox']), torch.tensor(self.state), resize_factor, x_patch_h)
                pred_boxes_ori = unwraped_topk_pred_boxes_raw.cpu()
                pred_boxes_ori = torch.cat([(pred_boxes_ori[:,2:]+pred_boxes_ori[:,:2])/2, pred_boxes_ori[:,2:]-pred_boxes_ori[:,:2]], dim=-1)
                pred_boxes_ori = torch.stack([torch.tensor(clip_box(self.map_box_back_fullsize(pred_boxes_ori_single.tolist(), x_patch_h), H, W, margin=10))for pred_boxes_ori_single in pred_boxes_ori], dim=0)
            self.state = clip_box(self.map_box_back_fullsize(unwraped_box[0].tolist(), x_patch_h), H, W, margin=10)
        
      
        if self.cfg.TEST.USE_REGION_ADJUST:
            self.flag = F.softmax(out_dict['logits'].cpu(), dim=1)[0][-1].item()
            if self.flag < self.T_logits:  
                if pred_score_map.max()< self.T_score: 
                    self.use_hann=False
                    self.search_factor = self.search_factor + self.step
                    if self.search_factor > self.max_search_area_factor:
                        self.search_factor = self.max_search_area_factor
                else:   
                    self.search_factor = self.base_factor
                    self.use_hann=True
            else: 
                self.search_factor = self.base_factor  
                self.use_hann=True
                
        
        if self.debug:
            if not self.use_visdom:
                # green for gt
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                x2, y2, w2, h2 = info['gt_bbox'].tolist()
                cv2.rectangle(image_BGR, (int(x2),int(y2)), (int(x2+w2),int(y2+h2)), color=(0, 255, 0), thickness=2)
                
                # red for base
                x3, y3, w3, h3 = PRED_BB[self.frame_id]
                cv2.rectangle(image_BGR, (int(x3),int(y3)), (int(x3+w3),int(y3+h3)), color=(0,0,255), thickness=2)
                
                # blue for pred
                x1, y1, w1, h1 = self.state
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w1),int(y1+h1)), color=(255,0,0), thickness=2)
                
                cv2.putText(image_BGR, 'GroundTruth', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(image_BGR, 'FocusTrack', (250, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(image_BGR, 'Base', (510, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
                
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state, PRED_BB[self.frame_id], 
                                       self.frame_id, self.search_factor, self.flag), 'Tracking', 1, 'Tracking')
                
                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
                self.visdom.register(response.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'response')
                # self.visdom.register((image, self.state, *pred_boxes_ori[~((pred_boxes_ori == torch.tensor(self.state)).all(1))]), 'Tracking', 1, 'Tracking')

                
                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
                    
                if self.use_grid:
                    x_patch_arr_ori = img_tensor[0].permute(1,2,0).numpy()
                    x_patch_arr_resize = cv2.resize(x_patch_arr_ori, (self.params.search_size, self.params.search_size))
                    self.visdom.register(torch.from_numpy(x_patch_arr_resize).permute(2, 0, 1), 'image', 1, 'resized_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False 
                        break
                
                if 'full_mask' in out_dict:
                    search_size = self.cfg.DATA.SEARCH.SIZE * self.cfg.DATA.SEARCH.SIZE
                    search_mask = out_dict['full_mask'][:,-search_size: ]
                    if search_mask.dim() == 2:
                        n, hw = search_mask.size()
                        search_mask = search_mask.reshape(n, self.cfg.DATA.SEARCH.SIZE, self.cfg.DATA.SEARCH.SIZE)
    
                    mask = search_mask.permute(1, 2, 0).cpu().numpy()    
                    
                    
                    heatmap = cv2.applyColorMap(np.uint8(mask * 255), cv2.COLORMAP_JET)
                    x_patch = x_patch_arr
                    alpha=0.3
                    overlay = cv2.addWeighted(x_patch, 1 - alpha, heatmap, alpha, 0)
                    show_box = box_cxcywh_to_xywh(torch.tensor(pred_boxes.mean(dim=0))) * self.cfg.TEST.SEARCH.SIZE       
                    cv2.rectangle(overlay,
                                    (int(show_box[0]), int(show_box[1])),
                                    (int(show_box[0] + show_box[2]), int(show_box[1] + show_box[3])), (255, 0, 0), 2)
                    self.visdom.register(torch.from_numpy(overlay).permute(2, 0, 1), 'image', 1, 'full_mask')
                    
                    
                    self.visdom.register(out_dict['feat_mask'].view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'feat_mask')
        

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}
    
    def compute_iou(self, box1, box2):
        """ Compute IoU between two boxes in (x, y, width, height) format.
        args:
            box1: list or tensor of format [x, y, width, height]
            box2: list or tensor of format [x, y, width, height]
        returns:
            iou: float - intersection over union value
        """
        box1 = torch.tensor(box1)
        box2 = torch.tensor(box2)

        # Convert (x, y, width, height) to (x1, y1, x2, y2)
        box1_x1y1 = box1[0:2]
        box1_x2y2 = box1[0:2] + box1[2:4]
        box2_x1y1 = box2[0:2]
        box2_x2y2 = box2[0:2] + box2[2:4]

        # Calculate intersection
        inter_x1 = torch.max(box1_x1y1[0], box2_x1y1[0])
        inter_y1 = torch.max(box1_x1y1[1], box2_x1y1[1])
        inter_x2 = torch.min(box1_x2y2[0], box2_x2y2[0])
        inter_y2 = torch.min(box1_x2y2[1], box2_x2y2[1])
        inter_area = torch.max(inter_x2 - inter_x1, torch.tensor(0.0)) * torch.max(inter_y2 - inter_y1, torch.tensor(0.0))

        # Calculate union
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area

        # Compute IoU
        iou = inter_area / union_area
        return iou.item()

    

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
    
    def map_box_back_fullsize(self, pred_box: list, full_size: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * full_size
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output)
            # 可以多注册几个
            )

        self.enc_attn_weights = enc_attn_weights

    def cal_box_from_coord(self, out_dict, coord,  resize_factor, H , W, reduce='mean'):
        """
        coord - (2) or (num_box, 2) dtype:long
        """
        pred_boxes= self.network.box_head.cal_bbox_from_idx(coord, out_dict['size_map'], out_dict['offset_map'])
        
        pred_boxes = pred_boxes.view(-1, 4)
        if reduce == 'mean':
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            return clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        elif reduce == 'none':
            boxes = []            
            pred_boxes = (pred_boxes * self.params.search_size / resize_factor).tolist()
            for box in pred_boxes:
                boxes.append(torch.tensor(clip_box(self.map_box_back(box, resize_factor), H, W, margin=10)))
            return torch.stack(boxes)
        elif reduce == 'raw':
            return pred_boxes
        else:
            raise NotImplementedError

def get_tracker_class():
    return FocusTrack
