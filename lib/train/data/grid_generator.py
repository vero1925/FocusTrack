import torch
import numpy as np
import torch.nn.functional as F
from cvxopt import matrix, solvers
from scipy.linalg import block_diag
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def unwarp_bboxes(bboxes, grid, output_shape):
    """Unwarps a tensor of bboxes of shape (n, 4) or (n, 5) according to the grid \
    of shape (h, w, 2) used to warp the corresponding image and the \
    output_shape (H, W, ...)."""
    bboxes = bboxes.clone()
    # image map of unwarped (x,y) coordinates
    img = grid.permute(2, 0, 1).unsqueeze(0)

    warped_height, warped_width = grid.shape[0:2]
    xgrid = 2 * (bboxes[:, 0:4:2] / warped_width) - 1
    ygrid = 2 * (bboxes[:, 1:4:2] / warped_height) - 1
    grid = torch.stack((xgrid, ygrid), dim=2).unsqueeze(0)

    # warped_bboxes has shape (2, num_bboxes, 2)
    warped_bboxes = F.grid_sample(
        img, grid, align_corners=True, padding_mode="border").squeeze(0)
    bboxes[:, 0:4:2] = (warped_bboxes[0] + 1) / 2 * output_shape[1]
    bboxes[:, 1:4:2] = (warped_bboxes[1] + 1) / 2 * output_shape[0]

    return bboxes

def unwarp_bboxes_batch(bboxes, grid):
    """
    args:
        bboxes: torch.Tensor (bs,n,4) xyxy [0,1]
        grid: torch.Tensor (bs, h, w,2)
        output_shape: tuple h,w
    """
    bboxes = bboxes.clone()
    # image map of unwarped (x,y) coordinates
    img = grid.permute(0, 3, 1, 2)

    xgrid = 2 * (bboxes[:, :, 0:4:2]) - 1
    ygrid = 2 * (bboxes[:, :, 1:4:2]) - 1
    grid = torch.stack((xgrid, ygrid), dim=3)

    warped_bboxes = F.grid_sample(
        img, grid, align_corners=True, padding_mode="border")

    bboxes[:, :, 0:4:2] = (warped_bboxes[:, 0,...] + 1) / 2 
    bboxes[:, :, 1:4:2] = (warped_bboxes[:, 1,...] + 1) / 2 

    return bboxes

def warp_boxes(bboxes, grid):
    """
    args:
        bboxes: torch.Tensor (bs,n,4) xyxy [0,1]
        grid: torch.Tensor (bs, h, w,2) [-1,1] xy
    returns:
        warpped_boxes: torch.Tensor (bs,n,4) xyxy [0,1]
    """
    bs, h, w, _ = grid.shape
    whwh = torch.tensor([w,h,w,h], device=grid.device)
    _, n, _ = bboxes.shape
    bboxes_grid_scale = bboxes.clone() * 2 - 1
    points = torch.cat([bboxes_grid_scale[...,:2], bboxes_grid_scale[...,2:]], dim=1) # (bs, 2n, 2)
    x_idx = torch.searchsorted(grid[...,0].contiguous(), points[:,None,:,0].expand(-1,h,-1).contiguous()) #(bs,h,2n)
    xs = torch.cat([grid[...,1], grid[:,:,-2:-1,1]], dim=-1).gather(dim=-1, index=x_idx) #(bs,h,2n)
    y_idx = torch.searchsorted(xs.permute(0,2,1).contiguous(), points[:,:,1:].contiguous()) #(bs, 2n, 1)
    x_idx_final = torch.cat([x_idx, x_idx[:,-1:,:]], dim=-2).gather(dim=1, index=y_idx.permute(0,2,1)).permute(0,2,1) #(bs, 2n, 1)
    y_idx_br = torch.where(y_idx>=h, h-1, y_idx) #(bs, 2n, 1)
    x_idx_br = torch.where(x_idx_final>=w, w-1, x_idx_final) #(bs, 2n, 1)
    y_idx_tl = torch.where(y_idx_br-1<=-1, 1, y_idx_br-1)
    x_idx_tl = torch.where(x_idx_br-1<=-1, 1, x_idx_br-1)
    idx_br = torch.cat([x_idx_br, y_idx_br], dim=-1) #(bs,2n,2) xy
    idx_tl = torch.cat([x_idx_tl, y_idx_tl], dim=-1) #(bs,2n,2) xy
    grid_point_idx = torch.stack([idx_tl, idx_br], dim=2) #(bs,2n,top2, 2) xy
    idx = grid_point_idx[...,1] * w + grid_point_idx[...,0] # (bs, 2n, top2)
    
    flatten_grid = grid.flatten(1,2).unsqueeze(1) #(bs, 1, hw, 2)
    grid_point = flatten_grid.expand(-1,2*n,-1,-1).gather(dim=-2, index=idx.unsqueeze(-1).expand(-1,-1,-1,2)) #(bs, 2n, topk, 2)
    p1 = grid_point[:,:,0,:]
    p2 = grid_point[:,:,1,:]
    p = points.squeeze(2)
    w1 = torch.div(p2-p, p2-p1)
    w2 = torch.div(p-p1, p2-p1)
    target = w1 * grid_point_idx[:,:,0,:] + w2 * grid_point_idx[:,:,1,:]
    out_box = torch.cat(target.split(n, dim=1), dim=-1)
    out_box_norm = out_box / whwh[None,None,:]
    return out_box_norm

class QPGrid():
    def __init__(self, 
                 grid_shape=(31, 51),
                 bandwidth_scale=1,
                 amplitude_scale=64,
                 grid_type='qp',
                 zoom_factor=1.5,
                 loss_dict={'asap':1},
                 constrain_type = 'none',
                 solver='cvxopt'
                 ):
        super().__init__()
        self.grid_shape = grid_shape
        self.bandwidth_scale = bandwidth_scale
        self.amplitude_scale = amplitude_scale
        self.grid_type = grid_type
        self.zoom_factor = zoom_factor
        self.loss_dict = loss_dict
        self.constrain_type = constrain_type
        self.solver = solver
        
        self.precompute_ind()

    def precompute_ind(self):
        m, n = self.grid_shape
        m -= 1
        n -= 1
        self.asap_k_row = np.concatenate([np.arange(m*n, dtype=int), np.arange(m*n, dtype=int)])
        self.asap_k_col = np.concatenate([np.arange(m, dtype=int).repeat(n), np.tile((np.arange(m, m+n, dtype=int)), m)])
        
        self.ts_r_row = np.arange(2*m*n)
        self.ts_r_col = np.concatenate([np.arange(m).repeat(n), np.tile((np.arange(n)+m), m)])
        
        self.ts_v_row = np.arange(2*m*n, dtype=int)
        self.ts_v_col = np.zeros(2*m*n, dtype=int)
        
        
        self.naive_constrain_A_np = np.stack([np.concatenate([np.ones(m), np.zeros(n)]), np.concatenate([np.zeros(m), np.ones(n)])])

    def bbox2sal(self, batch_bboxes, img_metas, jitter=None):
        """
        taken from https://github.com/tchittesh/fovea
        """
        h_out, w_out = self.grid_shape
        sals = []
        for i in range(len(img_metas)):
            h, w, _ = img_metas[i]['pad_shape']
            bboxes = batch_bboxes[i]
            if len(bboxes) == 0:  # zero detections case
                sal = np.ones((h_out, w_out)).expand_dims(0)
                sal /= sal.sum()
                sals.append(sal)
                continue
            
            if isinstance(batch_bboxes, torch.Tensor):
                if batch_bboxes.is_cuda:
                    bboxes = bboxes.cpu()
                bboxes = bboxes.numpy()
            cxy = bboxes[:, :2] + 0.5*bboxes[:, 2:]  # 中心点的坐标
            if jitter is not None:
                cxy += 2*jitter*(np.random.randn(*cxy.shape)-0.5)
            widths = (bboxes[:, 2] * self.bandwidth_scale).reshape(-1, 1)
            heights = (bboxes[:, 3] * self.bandwidth_scale).reshape(-1, 1)

            X, Y = np.meshgrid(
                np.linspace(0, w, w_out, dtype=np.float32),
                np.linspace(0, h, h_out, dtype=np.float32),
                indexing='ij'
            )
            grids = np.stack((X.flatten(), Y.flatten()), axis=1).T

            m, n = cxy.shape[0], grids.shape[1]

            norm1 = np.tile((cxy[:, 0:1]**2/widths + cxy[:, 1:2]**2/heights), (m, n))
            norm2 = grids[0:1, :]**2/widths + grids[1:2, :]**2/heights
            norms = norm1 + norm2

            cxy_norm = cxy
            cxy_norm[:, 0:1] /= widths
            cxy_norm[:, 1:2] /= heights

            distances = norms - 2*cxy_norm.dot(grids)

            sal = np.exp((-0.5 * distances))
            sal = self.amplitude_scale * (sal / (0.00001+sal.sum(axis=1, keepdims=True)))  # noqa: E501, normalize each distribution
            sal += 1/(self.grid_shape[0]*self.grid_shape[1])
            sal = sal.sum(axis=0)
            sal /= sal.sum()
            
            # scale saliency to peak==1
            sal = 1 / sal.max() * sal
            
            sal = sal.reshape(w_out, h_out).T[np.newaxis, ...]  # noqa: E501, add channel dimension
            sals.append(sal)
            
            
            # 保存 2D 和 3D 可视化图像
            # output_dir = '/data/users/wangying01/codes/antiuav/Two_Stage/two_stage_for_test/'         
            # # self.save_saliency_image(sal, i, output_dir, is_3d=False)  # 保存 2D 图像
            # self.save_saliency_image(sal, i, output_dir, is_3d=True)   # 保存 3D 图像
        return np.stack(sals)

    def bbox2sal_invert(self, batch_bboxes, img_metas, jitter=None):
        """
        Generate inverted saliency map: large values at edges, small values at the center.
        Compared with the original bbox2sal, this inverted saliency map is suitable for emphasizing the background area, 
        while still being able to use the non-uniform sampling grid generated by QPGrid.
        sal = sal.max() + sal.min() - sal
        sal /= sal.sum()
        sal = 1 / sal.max() * sal
        """
        h_out, w_out = self.grid_shape
        sals = []
        for i in range(len(img_metas)):
            h, w, _ = img_metas[i]['pad_shape']
            bboxes = batch_bboxes[i]
            if len(bboxes) == 0:  # zero detections case
                sal = np.ones((h_out, w_out))
                sal = np.expand_dims(sal, axis=0)
                sal /= sal.sum()
                sals.append(sal)
                continue
            
            if isinstance(batch_bboxes, torch.Tensor):
                if batch_bboxes.is_cuda:
                    bboxes = bboxes.cpu()
                bboxes = bboxes.numpy()
            cxy = bboxes[:, :2] + 0.5 * bboxes[:, 2:]  # Center coordinates
            if jitter is not None:
                cxy += 2 * jitter * (np.random.randn(*cxy.shape) - 0.5)
            widths = (bboxes[:, 2] * self.bandwidth_scale).reshape(-1, 1)
            heights = (bboxes[:, 3] * self.bandwidth_scale).reshape(-1, 1)

            X, Y = np.meshgrid(
                np.linspace(0, w, w_out, dtype=np.float32),
                np.linspace(0, h, h_out, dtype=np.float32),
                indexing='ij'
            )
            grids = np.stack((X.flatten(), Y.flatten()), axis=1).T

            m, n = cxy.shape[0], grids.shape[1]

            norm1 = np.tile((cxy[:, 0:1]**2 / widths + cxy[:, 1:2]**2 / heights), (m, n))
            norm2 = grids[0:1, :]**2 / widths + grids[1:2, :]**2 / heights
            norms = norm1 + norm2

            cxy_norm = cxy
            cxy_norm[:, 0:1] /= widths
            cxy_norm[:, 1:2] /= heights

            distances = norms - 2 * cxy_norm.dot(grids)

            # Compute saliency, then invert it (high values at edges)
            sal = np.exp((-0.5 * distances))
            sal = self.amplitude_scale * (sal / (0.00001 + sal.sum(axis=1, keepdims=True)))  # Normalize each distribution
            sal += 1 / (self.grid_shape[0] * self.grid_shape[1])
            sal = sal.sum(axis=0)
            
            # Invert the saliency
            sal = sal.max() + sal.min() - sal

            # Normalize to ensure sum is 1
            sal /= sal.sum()

            # Scale to peak == 1
            sal = 1 / sal.max() * sal

            sal = sal.reshape(w_out, h_out).T[np.newaxis, ...]  # Add channel dimension
            sals.append(sal)
            
            # 保存 2D 和 3D 可视化图像
            # output_dir = '/data/users/wangying01/codes/antiuav/Two_Stage/two_stage_for_test/'         
            # # self.save_saliency_image(sal, i, output_dir, is_3d=False)  # 保存 2D 图像
            # self.save_saliency_image(sal, i, output_dir, is_3d=True)   # 保存 3D 图像
            
            
            
        return np.stack(sals)
    
    def save_saliency_image(self, saliency, index, output_dir, is_3d=False):
        """
        保存 saliency 图像为 PNG 文件
        """
        saliency = saliency.squeeze()  # 去掉额外的维度，变成 2D 图像

        if is_3d:
            # 3D 可视化
            self.save_3d_saliency_image(saliency, index, output_dir)
        else:
            # 2D 可视化
            self.save_2d_saliency_image(saliency, index, output_dir)

    def save_2d_saliency_image(self, saliency, index, output_dir):
        """
        保存 2D saliency 图像为 PNG 文件
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(saliency, cmap='jet')
        plt.colorbar()
        plt.title(f"Saliency Map {index} (2D)")
        plt.axis('off')

        # 保存为 PNG 文件
        file_path = f"{output_dir}/saliency_map_{index}_2d.png"
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # 关闭图像，以释放内存

    def save_3d_saliency_image(self, saliency, index, output_dir):
        """
        保存 3D saliency 图像为 PNG 文件
        """
        # 创建 3D 坐标网格
        h_out, w_out = saliency.shape
        X, Y = np.meshgrid(np.arange(w_out), np.arange(h_out))

        # 创建 3D 图
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制 3D 表面图
        ax.plot_surface(X, Y, saliency, cmap='jet', edgecolor='none')
        ax.set_title(f"Saliency Map {index} (3D)")
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('Saliency')

        # 保存为 PNG 文件
        file_path = f"{output_dir}/saliency_map_{index}_3d.png"
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # 关闭图像，以释放内存
    
    def asap_loss(self, saliency, im_shape):
        m, n = saliency.shape
        h, w = im_shape
        mh = m/h
        nw = n/w
        k_left = (block_diag(*np.split(saliency, m ,axis=0)) * mh).T
        k_right = np.concatenate([-np.diag(saliency[i,:])*nw for i in range(m)], axis=0)
        k = np.concatenate([k_left, k_right], axis=1)
        P = 2 * np.matmul(k.T, k)
        q = np.zeros((m+n,1))
        P_mat = P
        q_mat = q
        return P_mat, q_mat
    
    def asap_ts(self, saliency, im_shape):
        m, n = saliency.shape
        h, w = im_shape
        saliency_sq = np.square(saliency)
        sal_x = saliency_sq.sum(axis=1)
        sal_y = saliency_sq.sum(axis=0)
        p_tl = np.diag(sal_x*(2*m*m/h/h+2))
        p_br = np.diag(sal_y*(2*n*n/w/w+2))
        p_tr = saliency_sq*(-2*m*n/h/w)
        P = np.block([
            [p_tl, p_tr],
            [p_tr.T, p_br]
        ])
        q = np.concatenate([sal_x*(-2*h/m/self.zoom_factor), sal_y*(-2*w/n/self.zoom_factor)])[:,None]
        return P, q
    
    def asap_ts_weight(self, saliency, im_shape, loss_list):
        """
        loss_list: list [ts_weight, asap_weight]
        """
        m, n = saliency.shape
        h, w = im_shape
        w_ts, w_asap = loss_list
        saliency_sq = np.square(saliency)
        sal_x = saliency_sq.sum(axis=1)
        sal_y = saliency_sq.sum(axis=0)
        p_tl = np.diag(sal_x*(2*w_asap*m*m/h/h+2*w_ts))
        p_br = np.diag(sal_y*(2*w_asap*n*n/w/w+2*w_ts))
        p_tr = saliency_sq*(-2*m*n/h/w*w_asap)
        P = np.block([
            [p_tl, p_tr],
            [p_tr.T, p_br]
        ])
        q = np.concatenate([sal_x*(-2*h/m/self.zoom_factor*w_ts), sal_y*(-2*w/n/self.zoom_factor*w_ts)])[:,None]
        return P, q    
    
    
    def ts_loss(self, saliency, im_shape):
        m, n = saliency.shape
        h, w = im_shape
        ideal_h =  h / m / self.zoom_factor
        ideal_w =  w / n / self.zoom_factor
        r_top = np.concatenate([(block_diag(*np.split(saliency, m ,axis=0)) ).T, np.zeros((m*n, n))], axis=1) 
        r_bottom = np.concatenate([np.zeros((m*n, m)), np.concatenate([np.diag(saliency[i,:]) for i in range(m)], axis=0)], axis=1) 
        v = np.concatenate([saliency[i,:] for i in range(m)])[:,None]
        R = np.concatenate([r_top, r_bottom], axis=0)
        V = np.concatenate([v*ideal_h, v*ideal_w], axis=0)
        P = 2 * np.matmul(R.T, R)
        q = -2 * np.matmul(R.T, V)
        P_mat = P
        q_mat = q
        return P_mat, q_mat
    
    
    def naive_constarin(self, saliency, im_shape):
        h, w = im_shape
        
        A = self.naive_constrain_A_np
        b = np.array([h, w])
        
        return A, b, None, None
    
    def get_qp_loss(self, saliency, im_shape):
        """
        args:
            saliency: np.ndarray (m,n)
            im_shape: shape of original image (h,w)
        returns:
            P, q, A, b, G, h
            min 1/2 * x^TPx + q^Tx
            s.t. Gx <= h
                 Ax = b
        """
        loss_func = {
            'asap': self.asap_loss,
            'ts': self.ts_loss,
            'asap_ts': self.asap_ts,
            'asap_ts_weight': self.asap_ts_weight
        }
        constrain_func = {
            'none': self.naive_constarin,
        }
        P_mat = 0
        q_mat = 0
        for loss_name, weight in self.loss_dict.items():
            if loss_name == 'asap_ts_weight':
                P_mat_single, q_mat_single = loss_func[loss_name](saliency, im_shape, weight)
                P_mat += P_mat_single
                q_mat += q_mat_single
            else:
                P_mat_single, q_mat_single = loss_func[loss_name](saliency, im_shape)
                P_mat += weight * P_mat_single
                q_mat += weight * q_mat_single
        A_mat, b_mat, G_mat, h_mat = constrain_func[self.constrain_type](saliency, im_shape)
        return P_mat, q_mat, A_mat, b_mat, G_mat, h_mat

    def cvxopt(self, P, q, A, b, G, h):
        P_mat = matrix(np.float64(P))
        q_mat = matrix(np.float64(q))
        A_mat = matrix(np.float64(A))
        b_mat = matrix(np.float64(b))
        sol = solvers.qp(P=P_mat,q=q_mat,A=A_mat,b=b_mat)
        x_array = np.array(sol['x'])
        return torch.tensor(x_array,dtype=torch.float32)

    def solve(self, P, q, A, b, G, h):
        """
        args:
            all input should be np.ndarry, np.float32
        returns:
            torch.Tensor torch.float32
        """
        solver_fn = {
            'cvxopt': self.cvxopt,
        }
        return solver_fn[self.solver](P, q, A, b, G, h)
    
    def qp_grid(self, img, saliency, out_shape, **kwargs):
        """
        img: (bs, channel, h, w)
        saliency: (bs, h, w)
        """
        bs, m, n = saliency.shape
        m -= 1
        n -= 1
        grid_list = []
        for i, sal in enumerate(saliency):
            sal_center = (sal[:-1,:-1] + sal[1:,1:] + sal[:-1,1:] + sal[1:,:-1]) / 4  # (16,16)
            h, w = img[i].shape[1:]
            P_mat, q_mat, A_mat, b_mat, G_mat, h_mat = self.get_qp_loss(sal_center, img[i, 0].shape)
            x = self.solve(P_mat, q_mat, A_mat, b_mat, G_mat, h_mat)   # (32,1)


            ygrid = torch.cat([torch.zeros(1,1),torch.cumsum(x[:m], 0)], dim=0) / h
            ygrid = torch.clamp(ygrid*2-1, min=-1, max=1) # 将grid的值从[0,1]映射到[-1,1]，pytorch的grid_sample要求输出坐标在[-1,1]
            ygrid = ygrid.view(-1, 1, self.grid_shape[0], 1)
            ygrid = ygrid.expand(-1, 1, *self.grid_shape)

            xgrid = torch.cat([torch.zeros(1,1),torch.cumsum(x[-n:], 0)], dim=0) / w
            xgrid = torch.clamp(xgrid*2-1, min=-1, max=1)
            xgrid = xgrid.view(-1, 1, 1, self.grid_shape[1])
            xgrid = xgrid.expand(-1, 1, *self.grid_shape)
            
            grid = torch.cat((xgrid, ygrid), 1)
            grid_list.append(grid)
            
        
        grids = torch.cat(grid_list, dim=0)   # torch.Size([1, 2, 17, 17])
        grids = F.interpolate(grids, size=out_shape, mode='bilinear',
                             align_corners=True)     # torch.Size([1, 2, 256, 256])
        return grids.permute(0, 2, 3, 1)  # torch.Size([1, 256, 256, 2])

    def gen_grid_from_saliency(self, img, saliency, out_shape, **kwargs):
        grid_func = {
            'qp': self.qp_grid
        }
        return grid_func[self.grid_type](img, saliency, out_shape, **kwargs)

    def forward(self, imgs, img_metas, gt_bboxes, out_shape, jitter=None, mode='center'):
        """
        args:
            imgs: torch.Tensor (bs, channel, h, w)
            img_metas: list of dict, len==bs, dict containing:
                        pad_shape: tuple, (h, w, c)
            gt_bboxes: torch.Tensor (bs, num_box, 4), dtype: float32
        returns:

        """
        if isinstance(gt_bboxes, torch.Tensor):
            batch_bboxes = gt_bboxes
        else:
            if len(gt_bboxes[0].shape) == 3:
                batch_bboxes = gt_bboxes[0].clone()  # noqa: E501, removing the augmentation dimension
            else:
                batch_bboxes = [bboxes.clone() for bboxes in gt_bboxes]
        device = batch_bboxes[0].device
        saliency = self.bbox2sal(batch_bboxes, img_metas)
        
        if mode == 'center':
            saliency = self.bbox2sal(batch_bboxes, img_metas)
        elif mode == 'margin':
            saliency = self.bbox2sal_invert(batch_bboxes, img_metas)
        else:
            raise ValueError('mode should be center or margin')

        # output_dir = '/data/users/wangying01/codes/antiuav/ZoomTrack/'
        # self.overlay_saliency_on_image(imgs, saliency, img_metas, output_dir, output_type='3d')
        grid = self.gen_grid_from_saliency(imgs, np.squeeze(saliency, axis=1), out_shape)
        
        return grid.to(device), saliency
    
    
    def overlay_saliency_on_image(self, imgs, saliency, img_metas, output_dir, output_type='2d'):
        """
        将 saliency 图像叠加到原始图像上并保存为 PNG 或 3D 图像
        :param output_type: '2d' 或 '3d'，决定是保存为 2D 叠加图像还是 3D 图像
        """
        for i in range(len(img_metas)):
            img = imgs[i].cpu().numpy().transpose(1, 2, 0)  # (h, w, c)
            img_meta = img_metas[i]
            h, w, _ = img_meta['pad_shape']
            sal = saliency[i].squeeze()  # 去掉额外的维度

            # 重新调整 saliency 的尺寸以匹配原始图像
            sal_resized = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)

            # 对 saliency 进行归一化处理
            sal_resized = (sal_resized - sal_resized.min()) / (sal_resized.max() - sal_resized.min())  # 归一化到[0, 1]
            
            # 使用 jet colormap映射 saliency，确保映射后的值在 [0, 1] 范围内
            sal_resized = plt.cm.jet(sal_resized)[:, :, :3]  # 使用 jet colormap，获取 RGB 部分

            # 将 saliency 图像叠加到原图上
            alpha = 0.5  # 透明度
            overlay = (alpha * sal_resized + (1 - alpha) * img)  # 叠加图像

            # 确保 overlay 图像的值在 [0, 1] 范围内
            overlay = np.clip(overlay, 0, 1)

            if output_type == '2d':
                # 保存为 PNG 文件
                file_path = f"{output_dir}/saliency_overlay_{i}_2d.png"
                plt.imsave(file_path, overlay)

            elif output_type == '3d':
                # 生成 X, Y 网格
                X, Y = np.meshgrid(np.arange(sal_resized.shape[1]), np.arange(sal_resized.shape[0]))

                # 将 Z 设置为 sal_resized 确保 Z 是二维的
                Z = sal_resized  # 确保 Z 是二维数组

                # 创建 3D 图形
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # 使用 plot_surface 绘制 3D 表面图
                ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=overlay, shade=True)

                # 设置标签和标题
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Saliency')
                ax.set_title(f"Saliency Overlay {i} (3D)")

                # 保存为 3D 图像文件
                file_path = f"{output_dir}/saliency_overlay_{i}_3d.png"
                plt.savefig(file_path)
                plt.close(fig)
    
    
    # def overlay_saliency_on_image(self, imgs, saliency, img_metas, output_dir):
    #     """
    #     将 saliency 图像叠加到原始图像上并保存为 PNG 文件 (2D 版本)
    #     """
    #     for i in range(len(img_metas)):
    #         img = imgs[i].cpu().numpy().transpose(1, 2, 0)  # (h, w, c)
    #         img_meta = img_metas[i]
    #         h, w, _ = img_meta['pad_shape']
    #         sal = saliency[i].squeeze()  # 去掉额外的维度

    #         # 重新调整 saliency 的尺寸以匹配原始图像
    #         sal_resized = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)

    #         # 对 saliency 进行归一化处理并映射为颜色
    #         sal_resized = (sal_resized - sal_resized.min()) / (sal_resized.max() - sal_resized.min())  # 归一化
    #         sal_resized = plt.cm.jet(sal_resized)[:, :, :3]  # 使用 jet colormap

    #         # 将 saliency 图像叠加到原图上
    #         alpha = 0.5  # 透明度
    #         overlay = (alpha * sal_resized + (1 - alpha) * img)  # 叠加图像

    #         # 保存为 PNG 文件
    #         file_path = f"{output_dir}/saliency_overlay_{i}_2d.png"
    #         plt.imsave(file_path, overlay)
    
    
    

    def save_overlay_2d(self, img, saliency, index, output_dir):
        """
        保存 2D 叠加图像为 PNG 文件
        """
        sal = (saliency - saliency.min()) / (saliency.max() - saliency.min())  # 归一化
        sal = plt.cm.jet(sal)[:, :, :3]  # 使用 jet colormap

        # 将 saliency 图像叠加到原图上
        alpha = 0.5  # 透明度
        overlay = (alpha * sal + (1 - alpha) * img)  # 叠加图像

        # 保存为 PNG 文件
        file_path = f"{output_dir}/saliency_overlay_{index}_2d.png"
        plt.imsave(file_path, overlay)


    def save_overlay_3d(self, img, saliency, index, output_dir, h, w):
        """
        保存 3D 叠加图像为 PNG 文件
        """
        # 创建 3D 坐标网格
        X, Y = np.meshgrid(np.arange(w), np.arange(h))

        # 创建 3D 图
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制 3D 图像表面
        ax.plot_surface(X, Y, saliency, cmap='jet', edgecolor='none', alpha=0.5)

        # 绘制原图作为背景
        ax.imshow(img, aspect='auto', extent=(0, w, h, 0), alpha=0.5)

        # 保存为 PNG 文件
        file_path = f"{output_dir}/saliency_overlay_{index}_3d.png"
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    
def build_data_search_grid_generator(cfg):
    loss_dict = dict()
    for loss_name, loss_weight in zip(cfg.DATA.SEARCH.GRID.GENERATOR.LOSS.NAMES, cfg.DATA.SEARCH.GRID.GENERATOR.LOSS.WEIGHTS):
        loss_dict[loss_name] = loss_weight
    grid_generator = QPGrid(
        grid_shape=cfg.DATA.SEARCH.GRID.SHAPE,
        bandwidth_scale=cfg.DATA.SEARCH.GRID.GENERATOR.BANDWIDTH_SCALE,
        amplitude_scale=1,
        zoom_factor=cfg.DATA.SEARCH.GRID.GENERATOR.ZOOM_FACTOR,
        grid_type=cfg.DATA.SEARCH.GRID.TYPE,
        loss_dict = loss_dict
    )
    return grid_generator