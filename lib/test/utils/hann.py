import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt


def hann1d(sz: int, centered = True) -> torch.Tensor:
    """1D cosine window."""
    if centered:
        return 0.5 * (1 - torch.cos((2 * math.pi / (sz + 1)) * torch.arange(1, sz + 1).float()))
    w = 0.5 * (1 + torch.cos((2 * math.pi / (sz + 2)) * torch.arange(0, sz//2 + 1).float()))
    return torch.cat([w, w[1:sz-sz//2].flip((0,))])


def hann2d(sz: torch.Tensor, centered = True) -> torch.Tensor:
    """2D cosine window."""
    return hann1d(sz[0].item(), centered).reshape(1, 1, -1, 1) * hann1d(sz[1].item(), centered).reshape(1, 1, 1, -1)


# def hann1d_new(sz: int, centered=True, sharpness=1.0) -> torch.Tensor:
#     """
#     1D cosine window with adjustable sharpness.
    
#     Args:
#         sz (int): Size of the window.
#         centered (bool): Whether the window is symmetric or not.
#         sharpness (float): Controls the sharpness of the window. 
#                            Smaller values make the window sharper (steeper difference), 
#                            larger values make it smoother.
#     Returns:
#         torch.Tensor: 1D cosine window.
#     """
#     if sharpness <= 0:
#         raise ValueError("sharpness must be positive")
    
#     # Adjust the cosine term using sharpness
#     if centered:
#         window = 0.5 * (1 - torch.cos((2 * math.pi / (sz + 1)) * torch.arange(1, sz + 1).float()))
#     else:
#         w = 0.5 * (1 + torch.cos((2 * math.pi / (sz + 2)) * torch.arange(0, sz // 2 + 1).float()))
#         window = torch.cat([w, w[1:sz - sz // 2].flip((0,))])
    
    
#     # Apply sharpness adjustment: scale the deviaiton from the mean
#     mean_val = window.mean()
#     window = mean_val + (window - mean_val) * (1 / sharpness)
    
#     # Normalize to maintain original mean and variance
#     original_mean = 0.5
#     original_std = (0.5 * (1 - torch.cos(torch.linspace(0, math.pi, sz))).std())
    
#     window -= window.mean()  # Zero mean
#     window /= window.std()  # Unit variance
#     window *= original_std  # Match original std
#     window += original_mean  # Match original mean

#     # Ensure all values are non-negative
#     window = torch.clamp(window, min=0)  # Ensure no values fall below 0
    
    
#     if sharpness!=1.0:
#         print('111')
#         output_file = "hann1d_sharpness_comparison.png"
        
#         plt.figure(figsize=(8, 6))
#         plt.plot(window.numpy(), label=f"Sharpness={sharpness}")
#         plt.legend()
#         plt.title("Hann1d with Adjustable Sharpness")
#         plt.savefig(output_file, dpi=300)
#         plt.close()
    
#     return window


# def hann2d(sz: torch.Tensor, sharpness, centered = True) -> torch.Tensor:
#     """2D cosine window."""
#     return hann1d_new(sz[0].item(), centered, sharpness).reshape(1, 1, -1, 1) \
#          * hann1d_new(sz[1].item(), centered, sharpness).reshape(1, 1, 1, -1)




def hann2d_bias(sz: torch.Tensor, ctr_point: torch.Tensor, centered = True) -> torch.Tensor:
    """2D cosine window."""
    distance = torch.stack([ctr_point, sz-ctr_point], dim=0)
    max_distance, _ = distance.max(dim=0)

    hann1d_x = hann1d(max_distance[0].item() * 2, centered)
    hann1d_x = hann1d_x[max_distance[0] - distance[0, 0]: max_distance[0] + distance[1, 0]]
    hann1d_y = hann1d(max_distance[1].item() * 2, centered)
    hann1d_y = hann1d_y[max_distance[1] - distance[0, 1]: max_distance[1] + distance[1, 1]]

    return hann1d_y.reshape(1, 1, -1, 1) * hann1d_x.reshape(1, 1, 1, -1)



def hann2d_clipped(sz: torch.Tensor, effective_sz: torch.Tensor, centered = True) -> torch.Tensor:
    """1D clipped cosine window."""

    # Ensure that the difference is even
    effective_sz += (effective_sz - sz) % 2
    effective_window = hann1d(effective_sz[0].item(), True).reshape(1, 1, -1, 1) * hann1d(effective_sz[1].item(), True).reshape(1, 1, 1, -1)

    pad = (sz - effective_sz) // 2

    window = F.pad(effective_window, (pad[1].item(), pad[1].item(), pad[0].item(), pad[0].item()), 'replicate')

    if centered:
        return window
    else:
        mid = (sz / 2).int()
        window_shift_lr = torch.cat((window[:, :, :, mid[1]:], window[:, :, :, :mid[1]]), 3)
        return torch.cat((window_shift_lr[:, :, mid[0]:, :], window_shift_lr[:, :, :mid[0], :]), 2)


def gauss_fourier(sz: int, sigma: float, half: bool = False) -> torch.Tensor:
    if half:
        k = torch.arange(0, int(sz/2+1))
    else:
        k = torch.arange(-int((sz-1)/2), int(sz/2+1))
    return (math.sqrt(2*math.pi) * sigma / sz) * torch.exp(-2 * (math.pi * sigma * k.float() / sz)**2)


def gauss_spatial(sz, sigma, center=0, end_pad=0):
    k = torch.arange(-(sz-1)/2, (sz+1)/2+end_pad)
    return torch.exp(-1.0/(2*sigma**2) * (k - center)**2)


def label_function(sz: torch.Tensor, sigma: torch.Tensor):
    return gauss_fourier(sz[0].item(), sigma[0].item()).reshape(1, 1, -1, 1) * gauss_fourier(sz[1].item(), sigma[1].item(), True).reshape(1, 1, 1, -1)

def label_function_spatial(sz: torch.Tensor, sigma: torch.Tensor, center: torch.Tensor = torch.zeros(2), end_pad: torch.Tensor = torch.zeros(2)):
    """The origin is in the middle of the image."""
    return gauss_spatial(sz[0].item(), sigma[0].item(), center[0], end_pad[0].item()).reshape(1, 1, -1, 1) * \
           gauss_spatial(sz[1].item(), sigma[1].item(), center[1], end_pad[1].item()).reshape(1, 1, 1, -1)


def cubic_spline_fourier(f, a):
    """The continuous Fourier transform of a cubic spline kernel."""

    bf = (6*(1 - torch.cos(2 * math.pi * f)) + 3*a*(1 - torch.cos(4 * math.pi * f))
           - (6 + 8*a)*math.pi*f*torch.sin(2 * math.pi * f) - 2*a*math.pi*f*torch.sin(4 * math.pi * f)) \
         / (4 * math.pi**4 * f**4)

    bf[f == 0] = 1

    return bf

def max2d(a: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Computes maximum and argmax in the last two dimensions."""

    max_val_row, argmax_row = torch.max(a, dim=-2)
    max_val, argmax_col = torch.max(max_val_row, dim=-1)
    argmax_row = argmax_row.view(argmax_col.numel(),-1)[torch.arange(argmax_col.numel()), argmax_col.view(-1)]
    argmax_row = argmax_row.reshape(argmax_col.shape)
    argmax = torch.cat((argmax_row.unsqueeze(-1), argmax_col.unsqueeze(-1)), -1)
    return max_val, argmax
