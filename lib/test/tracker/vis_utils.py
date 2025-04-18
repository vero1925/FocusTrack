
import torch
import numpy as np
from lib.utils.misc import NestedTensor
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import cv2
import os


############## used for visulize eliminated tokens #################
def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices


def gen_masked_tokens(tokens, indices, alpha=0.2):
    # indices = [i for i in range(196) if i not in indices]
    indices = indices[0].astype(int)
    tokens = tokens.copy()
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    return tokens


def recover_image(tokens, H, W, Hp, Wp, patch_size):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(Hp, Wp, patch_size, patch_size, 3).swapaxes(1, 2).reshape(H, W, 3)
    return image


def pad_img(img):
    height, width, channels = img.shape
    im_bg = np.ones((height, width + 8, channels)) * 255
    im_bg[0:height, 0:width, :] = img
    return im_bg


def gen_visualization(image, mask_indices, patch_size=16):
    # image [224, 224, 3]
    # mask_indices, list of masked token indices

    # mask mask_indices need to cat
    # mask_indices = mask_indices[::-1]
    num_stages = len(mask_indices)
    for i in range(1, num_stages):
        mask_indices[i] = np.concatenate([mask_indices[i-1], mask_indices[i]], axis=1)

    # keep_indices = get_keep_indices(decisions)
    image = np.asarray(image)
    H, W, C = image.shape
    Hp, Wp = H // patch_size, W // patch_size
    image_tokens = image.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)

    stages = [
        recover_image(gen_masked_tokens(image_tokens, mask_indices[i]), H, W, Hp, Wp, patch_size)
        for i in range(num_stages)
    ]
    imgs = [image] + stages
    imgs = [pad_img(img) for img in imgs]
    viz = np.concatenate(imgs, axis=1)
    return viz




def vis_feature_maps(features, resolution, img, save_path='.'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shape_f = [resolution, resolution]
    w, h, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    featuretype = "t" if resolution == 8 else "s"

    features_mean = []
    for feature in features:
        feature = feature[:, 1:, :]
        if resolution == 8:
            feature = feature[:, :64, :]
        elif resolution == 16:
            feature = feature[:, 64:, :]
        features_mean.append(feature.mean(dim=2).squeeze().reshape(shape_f).cpu().numpy())

    block_num = 0
    for feature in features_mean:
        feature = np.maximum(feature, 0)
        feature /= np.max(feature)
        # feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
        feature = cv2.resize(feature, (w, h))
        feature = np.uint8(255 * feature)
        feature = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
        figure = cv2.addWeighted(img, 1, feature, 0.5, 0)
        cv2.imwrite(save_path + '/' + featuretype + '_Block{}_feature.png'.format(block_num), figure)
        block_num += 1

    del features_mean


def vis_attn_maps(attn_weights, q_w, k_w, skip_len1, skip_len2, x1, x2, x1_title, x2_title, save_path='.', idxs=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shape1 = [q_w, q_w]
    shape2 = [k_w, k_w]

    attn_weights_mean = []
    for attn in attn_weights:
        attn_weights_mean.append(attn[..., skip_len1:(skip_len1+q_w**2), skip_len2:(skip_len2+k_w**2)].mean(dim=1).squeeze().reshape(shape1+shape2).cpu())

    # downsampling factor
    fact = 16

    if idxs is None:
        idxs = [(128, 128)]

    block_num = 0
    idx_o = idxs[0]
    for attn_weight in attn_weights_mean:
        fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        ax = fig.add_subplot(111)
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        ax.imshow(attn_weight[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
        ax.axis('off')
        # ax.set_title(f'Stage2-Block{block_num}')
        plt.savefig(save_path + '/Block{}_attn_weight.png'.format(block_num))
        plt.close()
        block_num += 1

    fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    x2_ax = fig.add_subplot(111)
    x2_ax.imshow(x2)
    x2_ax.axis('off')
    plt.savefig(save_path + '/{}.png'.format(x2_title))
    plt.close()

    # the reference points as red circles
    fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    x1_ax = fig.add_subplot(111)
    x1_ax.imshow(x1)
    for (y, x) in idxs:
        # scale = im.height / img.shape[-2]
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        x1_ax.add_patch(plt.Circle((x, y), fact // 2, color='r'))
        # x1_ax.set_title(x1_title)
        x1_ax.axis('off')
    plt.savefig(save_path+'/{}.png'.format(x1_title))
    plt.close()

    del attn_weights_mean




def compute_distance_matrix(patch_size, num_patches, length):
    """Helper function to compute distance matrix."""

    distance_matrix = np.zeros((num_patches, num_patches))

    for i in range(num_patches):
        for j in range(num_patches):
            if i == j: # zero distance
                continue

            xi, yi = (int(i/length)), (i % length)
            xj, yj = (int(j/length)), (j % length)

            distance_matrix[i, j] = patch_size*np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix



def compute_mean_attention_dist(patch_size, attention_weights):
        
    # num_cls_tokens = 2 if "distilled" in model_type else 1
    # The attention_weights shape = (batch, num_heads, num_patches, num_patches)
    # attention_weights = attention_weights[..., num_cls_tokens:, num_cls_tokens:]  # Removing the CLS token
    
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    assert length**2 == num_patches, "Num patches is not perfect square"

    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)   # patch_size的作用？
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    # The attention_weights along the last axis adds to 1 this is due to the fact that they are softmax of the raw logits
    # summation of the (attention_weights * distance_matrix) should result in an average distance per token.
    # 现在的情况是 attention_weights 最后一行之和不是1
    # 之后再看这个问题
    
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(mean_distances, axis=-1)  # Sum along last axis to get average distance per token
    mean_distances = np.mean(mean_distances, axis=-1)  # Now average across all the tokens

    return mean_distances



def vis_attn_distance(attn_weights, temp_start_idx, num_temp_token, search_start_idx, num_search_token,patch_size, save_path='.'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 对 attention 进行截断
    temp_attention=[]
    search_attention=[]
    for i in range(len(attn_weights)):
        temp_attention.append(attn_weights[i][..., temp_start_idx:temp_start_idx+num_temp_token, temp_start_idx:temp_start_idx+num_temp_token])
        search_attention.append(attn_weights[i][..., search_start_idx:search_start_idx+num_search_token, search_start_idx:search_start_idx+num_search_token])
    
    
    attention_score_dict_temp = {f"layer_{i}_temp_att_map": attention_weight
                            for i, attention_weight in enumerate(temp_attention)}   # 按层划分
    attention_score_dict_search = {f"layer_{i}_search_att_map": attention_weight
                            for i, attention_weight in enumerate(search_attention)}
    
    
    # attention distance -- temp
    mean_distances_temp = {f"{name}_mean_dist": compute_mean_attention_dist(
                    patch_size=16,
                    attention_weights=attention_weight.cpu().numpy(),)
                for name, attention_weight in attention_score_dict_temp.items() }   
    
    plt.figure(figsize=(9, 9))
    
    num_heads=12
    for idx in range(len(mean_distances_temp)):
        mean_distance = mean_distances_temp[f"layer_{idx}_temp_att_map_mean_dist"]
        x = [idx] * num_heads
        y = mean_distance[0, :]
        plt.scatter(x=x, y=y, label=f"transformer_block_{idx}")

    # plt.legend(loc="best")
    plt.xlabel("Attention Head", fontsize=14)
    plt.ylabel("Attention Distance", fontsize=14)
    plt.title("mean_distance_temp", fontsize=14)
    plt.grid()
    plt.savefig(os.path.join(save_path, 'mean_distance_temp.jpg'))
    
    print('stop.')


    # attention distance -- search
    mean_distances_search = {f"{name}_mean_dist": compute_mean_attention_dist(
                    patch_size=16,
                    attention_weights=attention_weight.cpu().numpy(),)
                for name, attention_weight in attention_score_dict_search.items() }
    
    plt.figure(figsize=(9, 9))
    
    num_heads=12
    for idx in range(len(mean_distances_search)):
        mean_distance = mean_distances_search[f"layer_{idx}_search_att_map_mean_dist"]
        x = [idx] * num_heads
        y = mean_distance[0, :]
        plt.scatter(x=x, y=y, label=f"transformer_block_{idx}")

    # plt.legend(loc="best")
    plt.xlabel("Attention Head", fontsize=14)
    plt.ylabel("Attention Distance", fontsize=14)
    plt.title("mean_distance_search", fontsize=14)
    plt.grid()
    plt.savefig(os.path.join(save_path, 'mean_distance_search.jpg'))
    
    print('stop.')            
    

    # mean_distances = compute_mean_attention_dist(patch_size, attn_weights)

    # fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    # fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    # ax = fig.add_subplot(111)
    # ax.imshow(mean_distances, cmap='cividis', interpolation='nearest')
    # ax.axis('off')
    # plt.savefig(save_path + '/attn_distance.png')
    # plt.close()
    