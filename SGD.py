
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import gc
import tools as t
import time

def calculate_const(A_init, sigma, sigma_moves, alpha, beta):
    J = np.shape(A_init)[0] * np.shape(A_init)[1]
    return (
            J * torch.log(1 / (sigma * torch.sqrt(torch.tensor(2, dtype=torch.float32) * torch.pi))) +
            (2 * torch.log(1 / (2 * sigma_moves * torch.pi))) -
            torch.log(torch.lgamma(alpha).exp() * torch.lgamma(beta).exp() / torch.lgamma(alpha + beta).exp())
    )
def estimate_transformations(image_name, A_init, lr=0.001, epochs=50, batch_size=2, sigma=0.1, total_samples=50):
    """Optimized transformation estimation with mini-batch SGD."""
    A_est = torch.nn.Parameter(torch.tensor(A_init, dtype=torch.float32, requires_grad=True))
    img_size = A_init.shape[0]
    best_poses = [[0,0,0,1]] * total_samples

    sigma_moves = torch.tensor(0.05 * img_size, dtype=torch.float32)
    alpha = torch.tensor(2, dtype=torch.float32)
    beta = torch.tensor(5, dtype=torch.float32)
    Const_in_loss = calculate_const(A_init, sigma, sigma_moves, alpha, beta)
    optimizer = torch.optim.SGD([A_est], lr=lr, momentum=0.6)

    step_total_loss = 0
    total_loss = 0
    for e in range(epochs):
        print(e)
        if step_total_loss != 0 and abs(total_loss - step_total_loss) / abs(step_total_loss) < 1e-4:
            return A_est.detach()
        step_total_loss = total_loss
        total_loss = 0
        for step in range(int(total_samples / batch_size)):
            optimizer.zero_grad()
            batch_X = t.get_data_list(range(step*batch_size , (step+1)*batch_size), image_name)
            Pose_X = best_poses[step*batch_size : (step+1)*batch_size]

            loss, pose = loss_function(batch_X, A_est, sigma, Const_in_loss, e, img_size, sigma_moves, alpha, beta, Pose_X)
            for i in range(batch_size):
                if len(pose[i])>0:
                    best_poses[step*batch_size + i] = pose[i]
            loss.backward()
            optimizer.step()

            total_loss += (loss.item() / total_samples)

    return A_est.clone().detach()


def comput_PDF_Scale(x, alpha, beta):
    """Precompute a 2D probability density function (PDF)."""
    PDF = torch.log((x ** (alpha - 1)) * ((1 - x) ** (beta - 1)))
    return PDF

def comput_PDF_Moves(x, y, sigma_):
    """Precompute a 2D probability density function (PDF) using broadcasting."""
    X, Y = torch.meshgrid(x, y, indexing='ij')
    PDF = (X ** 2 + Y ** 2) / (2 * sigma_ ** 2)
    return PDF

def loss_function(X_list, A_est, sigma, Const_in_loss, step, img_size, sigma_moves, alpha, beta, best_poses):
    """Compute loss function using optimized transformations."""


    loss = torch.tensor(0.0, dtype=torch.float32, device=A_est.device)
    min_pose = []
    img_num = 0
    for I in X_list:
        [angles, linex, liney, lineScale, K, p] = t.Calculate_linespaces(best_poses[img_num], step, img_size, A_est)
        with torch.no_grad():
            PDF_Moves = comput_PDF_Moves(linex, liney, sigma_moves)
            PDF_Scale = comput_PDF_Scale(lineScale, alpha.item(), beta.item())
        Const_in_loss += (2 * torch.log(1 / torch.tensor(p, dtype=torch.float32)) +
                          torch.log(2 * torch.pi / torch.tensor(K, dtype=torch.float32)) +
                          torch.log(1 / torch.tensor(3, dtype=torch.float32))
                          )
        rotated_images = torch.stack([
            TF.rotate(A_est.unsqueeze(0), angle=float(angle * (180 / torch.pi)),
                      interpolation=TF.InterpolationMode.BILINEAR)
            for angle in angles
        ])
        min_value = float('inf')
        tmp_pose = []
        t2 = torch.as_tensor(I, dtype=torch.float32, device=A_est.device).unsqueeze(0)
        norm_diffs = []
        index_angle = 0
        for transformed_A in rotated_images:
            for x in linex:
                x_shift = int(torch.clamp(x - linex.min(), 0, PDF_Moves.shape[0] - 1).item())
                for y in liney:
                    y_shift = int(torch.clamp(y - liney.min(), 0, PDF_Moves.shape[1] - 1).item())
                    shifted_A = torch.roll(transformed_A, shifts=(int(x.item()), int(y.item())), dims=(1, 2))
                    for scale in lineScale:
                        scale_idx = int(torch.clamp(scale - lineScale.min(), 0, PDF_Scale.shape[0] - 1).item())
                        scaled_A = scale * shifted_A
                        norm_diff = torch.linalg.matrix_norm(t2 - scaled_A) ** 2
                        a = (-norm_diff / (2 * sigma ** 2)) - PDF_Moves[x_shift, y_shift] + PDF_Scale[scale_idx]
                        norm_diffs.append(a.clone())
                        if abs(a) < min_value:
                            min_value = abs(a)
                            tmp_pose = [angles[index_angle],x,y,scale]
            index_angle += 1
        if (norm_diffs == []):
            print()

        norm_diffs = torch.stack(norm_diffs)
        max_val = torch.max(norm_diffs)
        min_pose.append(tmp_pose)
        loss += ((torch.logsumexp(norm_diffs - max_val, dim=0).squeeze() - max_val) + Const_in_loss.squeeze())
        img_num += 1
    return [loss, min_pose]



if __name__ == '__main__':
    #image_name = input('Enter image name:')
    start = time.time()
    image_name = 'pikacho2'
    total_samples = 50
    data = t.get_data_list(range(total_samples), image_name)
    A_init = t.get_mean_img(total_samples, image_name)
    gc.collect()
    # lr = input("Enter learning rate: (Recommended value: (0.001-0.004) higher for less details images):")
    # epochs = input("Enter number of epochs: (Recommended 50):")
    # batch_size = input("Enter batch size: (Recommended 2 for small CPU):")
    lr = 0.0001
    A_est = estimate_transformations(image_name, A_init,lr = lr, epochs=100, batch_size= 1,total_samples = total_samples)

    figure, axis = plt.subplots(1, 2)
    A = A_est.squeeze(0)
    end = time.time()

    print(f"Runtime: {end - start:.4f} seconds")

    axis[0].set(title="Data example")
    axis[0].imshow(A_init, cmap='gray')
    axis[1].set(title=f"Output: {lr}")
    axis[1].imshow(A, cmap='gray')

    plt.show()
