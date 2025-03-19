import torch
import mrcfile
import numpy as np
def normalize(tensor):
    """Normalize a tensor to [0,1] range."""
    max =tensor.max()
    min = tensor.min()
    if max > 1 or min < 0:
        return (tensor - min) / (max - min)
    return tensor



def Calculate_linespaces (current_pose, step,img_size, A_est):
    pixels = int(0.05 * img_size)
    p = int(pixels / 2)
    lineScale = torch.linspace(0.8, 1.2, 3, device=A_est.device)
    if step == 0:
        K = 90
        angles = torch.linspace(0, 2 * torch.pi, K, device=A_est.device)
        linex = torch.linspace(-pixels, pixels, p, device=A_est.device).round().int()
        liney = torch.linspace(-pixels, pixels, p, device=A_est.device).round().int()

    elif step == 1:
        K = 45
        angles = torch.linspace(current_pose[0] - torch.pi/2, current_pose[0] + torch.pi/2, K, device=A_est.device)
        linex = torch.linspace(current_pose[1] - pixels / 2, current_pose[1] + pixels / 2, p, device=A_est.device).round().int()
        liney = torch.linspace(current_pose[2] - pixels / 2, current_pose[2] + pixels / 2, p, device=A_est.device).round().int()

    elif step<=3:
        K = 90
        p = int(p / 2)

        angles = torch.linspace(current_pose[0] - torch.pi/4, current_pose[0] + torch.pi/4, K, device=A_est.device)
        linex = torch.linspace(int(current_pose[1]-pixels/4), int(current_pose[1]+pixels/4), p, device=A_est.device).round().int()
        liney = torch.linspace(int(current_pose[2]-pixels/4), int(current_pose[2]+pixels/4), p, device=A_est.device).round().int()
        lineScale = torch.linspace(0.8, 1.2, 3, device=A_est.device)
    elif step <=8:
        K = 45
        p = int(p/2)
        angles = torch.linspace(current_pose[0] - torch.pi/8, current_pose[0] + torch.pi/8, K, device=A_est.device)
        linex = torch.linspace(int(current_pose[1]-pixels/8),int (current_pose[1]+pixels/8), p, device=A_est.device).round().int()
        liney = torch.linspace(int(current_pose[2]-pixels/8), int(current_pose[2]+pixels/8),  p, device=A_est.device).round().int()
    else:
        K = 22
        p = int(p / 2)
        angles = torch.linspace(current_pose[0] - torch.pi / 16, current_pose[0] + torch.pi / 16, K, device=    A_est.device)
        linex = torch.linspace(int(current_pose[1] - pixels / 16), int(current_pose[1] + pixels / 16), p,
                               device=A_est.device).round().int()
        liney = torch.linspace(int(current_pose[2] - pixels / 16), int(current_pose[2] + pixels / 16), p,
                               device=A_est.device).round().int()



    return [angles, linex, liney, lineScale, K, p]


def get_data_list(index_list, image_name):
    """Load and normalize items from the data."""
    arr = []
    for i in index_list:
        img = mrcfile.read(f'./output/{image_name + str(i)}.mrc')
        arr.append(normalize(img))

    return arr
def get_mean_img(total_samples, image_name):
    """create a mean of all images."""
    arr = []
    for i in range(total_samples):
        img = mrcfile.read(f'./output/{image_name + str(i)}.mrc')
        arr.append(normalize(img))
    stack = np.stack(arr, axis=0)
    return normalize(np.mean(stack, axis=0))