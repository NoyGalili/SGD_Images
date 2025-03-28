
import mrcfile
import numpy as np
import Config as C
import os
from scipy.ndimage import gaussian_filter

def normalize(tensor):
    """Normalize a tensor to [0,1] range."""
    max =tensor.max()
    min = tensor.min()
    if max > 1 or min < 0:
        return (tensor - min) / (max - min)
    return tensor


import torch


def Calculate_linespaces(current_pose, step, img_size, A_est):
    """
    Calculate parameters for line spacing based on the current step and pose.

    Args:
        current_pose (tuple): Contains values related to the current position and scale.
        step (int): The iteration step, which determines the refinement level.
        img_size (int): The size of the image (assumed to be square dimensions).
        A_est (torch.Tensor): A tensor used to determine the device for computations.

    Returns:
        list: A list containing angles, linex, liney, lineScale, K (number of angles), and p (number of positions).
    """
    pixels = int(C.pixel_range * img_size)  # Calculate the number of pixels (6% of image size)
    p = int(pixels / 2)  # Initial half pixel range
    lineScale = torch.linspace(0.8, 1.2, 3, device=A_est.device)  # Initial scaling range

    if step == 0:
        # Initial step: Full 360-degree search
        K = C.K_angles.K1.value
        angles = torch.linspace(0, 2 * torch.pi, K, device=A_est.device)
        linex = torch.linspace(-pixels, pixels, p, device=A_est.device).round().int()
        liney = torch.linspace(-pixels, pixels, p, device=A_est.device).round().int()

    elif step == 1:
        # Step 1: Reduce search to 180 degrees around the current angle
        K = C.K_angles.K1.value
        p = int(p / 2)  # Reduce pixel range
        angles = torch.linspace(current_pose[0] - torch.pi / 2, current_pose[0] + torch.pi / 2, K, device=A_est.device)
        linex = torch.linspace(current_pose[1] - pixels / 2, current_pose[1] + pixels / 2, p,
                               device=A_est.device).round().int()
        liney = torch.linspace(current_pose[2] - pixels / 2, current_pose[2] + pixels / 2, p,
                               device=A_est.device).round().int()
        lineScale = torch.linspace(current_pose[3] - 0.1, current_pose[3] + 0.1, 3, device=A_est.device)

    elif step <= 3:
        # Steps 2-3: Narrow angle and position range
        K = C.K_angles.K2.value
        p = int(p / 2)
        angles = torch.linspace(current_pose[0] - torch.pi / 4, current_pose[0] + torch.pi / 4, K, device=A_est.device)
        linex = torch.linspace(int(current_pose[1] - pixels / 4), int(current_pose[1] + pixels / 4), p,
                               device=A_est.device).round().int()
        liney = torch.linspace(int(current_pose[2] - pixels / 4), int(current_pose[2] + pixels / 4), p,
                               device=A_est.device).round().int()
        lineScale = torch.linspace(current_pose[3] - 0.07, current_pose[3] + 0.07, 3, device=A_est.device)

    elif step <= 8:
        # Steps 4-8: Further reduction in search space
        K = C.K_angles.K3.value
        p = int(p / 2)
        angles = torch.linspace(current_pose[0] - torch.pi / 8, current_pose[0] + torch.pi / 8, K, device=A_est.device)
        linex = torch.linspace(int(current_pose[1] - pixels / 4), int(current_pose[1] + pixels / 4), p,
                               device=A_est.device).round().int()
        liney = torch.linspace(int(current_pose[2] - pixels / 8), int(current_pose[2] + pixels / 4), p,
                               device=A_est.device).round().int()
        lineScale = torch.linspace(current_pose[3] - 0.05, current_pose[3] + 0.05, 3, device=A_est.device)

    elif step <= 25:
        # Steps 9-25: Even finer adjustments
        K = C.K_angles.K4.value
        p = int(p / 4)
        angles = torch.linspace(current_pose[0] - torch.pi / 16, current_pose[0] + torch.pi / 16, K,
                                device=A_est.device)
        linex = torch.linspace(int(current_pose[1] - pixels / 16), int(current_pose[1] + pixels / 16), p,
                               device=A_est.device).round().int()
        liney = torch.linspace(int(current_pose[2] - pixels / 16), int(current_pose[2] + pixels / 16), p,
                               device=A_est.device).round().int()
        lineScale = torch.linspace(current_pose[3] - 0.05, current_pose[3] + 0.05, 3, device=A_est.device)

    else:
        # Final step: Very fine adjustments
        K = C.K_angles.K5.value
        p = int(p / 4)
        angles = torch.linspace(current_pose[0] - torch.pi / 32, current_pose[0] + torch.pi / 32, K,
                                device=A_est.device)
        linex = torch.linspace(int(current_pose[1] - pixels / 16), int(current_pose[1] + pixels / 16), p,
                               device=A_est.device).round().int()
        liney = torch.linspace(int(current_pose[2] - pixels / 16), int(current_pose[2] + pixels / 16), p,
                               device=A_est.device).round().int()
        lineScale = torch.linspace(current_pose[3] - 0.025, current_pose[3] + 0.025, 3, device=A_est.device)

    return [angles, linex, liney, lineScale, K, p]


def get_data_list(index_list, folder_path):
    """Load and normalize selected images from the given folder."""
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.mrc')])
    arr = []
    for i in index_list:
        file_path = os.path.join(folder_path, file_list[i])
        img = mrcfile.read(file_path)
        arr.append(normalize(img))
    return arr

def get_mean_img(total_samples, folder_path):
    """Create a normalized mean image from the first N images in the folder."""
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.mrc')])
    arr = []
    for i in range(total_samples):
        file_path = os.path.join(folder_path, file_list[i])
        img = mrcfile.read(file_path)
        arr.append(normalize(img))
    stack = np.stack(arr, axis=0)
    return normalize(np.mean(stack, axis=0))


def estimate_noise_by_difference(image, sigma_blur=2):
    """
    Estimate noise by subtracting a Gaussian-smoothed image from the original.
    """
    smooth = gaussian_filter(image, sigma=sigma_blur)
    noise = image - smooth
    return np.std(noise)

def estimate_noise_batch(folder_path):
    noise_estimates = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.mrc'):
            img = mrcfile.read(os.path.join(folder_path, file))
            noise = estimate_noise_by_difference(img)
            noise_estimates.append(noise)
    return np.mean(noise_estimates)