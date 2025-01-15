import argparse
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    From https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html#formal-definition
"""
def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)



def load_tensors_to_matrix(directory_path):
    """
    Loads all .pt files in a directory and combines them into a matrix.
    
    Args:
        directory_path (str): Path to the directory containing .pt files.
        
    Returns:
        torch.Tensor: A matrix where each row corresponds to a tensor from a .pt file.
    """
    tensors = []
    
    try:
        for file_name in sorted(os.listdir(directory_path)):
            if file_name.endswith(".pt"):
                file_path = os.path.join(directory_path, file_name)
                tensor = torch.load(file_path)
                
                # Ensure tensor is 1D
                if tensor.ndim == 1:
                    tensors.append(tensor)
                else:
                    raise ValueError(f"Tensor in {file_path} is not 1D: {tensor.shape}")
        
        # Combine tensors into a matrix
        if tensors:
            return torch.stack(tensors)
        else:
            raise RuntimeError(f"No .pt files found in {directory_path}")
    
    except Exception as e:
        raise RuntimeError(f"Failed to load tensors from {directory_path}: {e}")


# Define your functions here, e.g., load_tensors_to_matrix and MMD

def main():
    parser = argparse.ArgumentParser(description="Compute MMD between two tensor directories.")
    parser.add_argument("path1", type=str, help="Path to the first directory containing .pt files.")
    parser.add_argument("path2", type=str, help="Path to the second directory containing .pt files.")
    parser.add_argument("--kernel", type=str, default="rbf", help="Kernel to use for MMD computation (default: rbf).")
    args = parser.parse_args()

    biomedclip_representations = load_tensors_to_matrix(args.path1)
    pmcoa2_intext_representations = load_tensors_to_matrix(args.path2)

    biomedclip_representations = biomedclip_representations.to(device)
    pmcoa2_intext_representations = pmcoa2_intext_representations.to(device)

    result = MMD(biomedclip_representations, pmcoa2_intext_representations, kernel=args.kernel)
    print(f"MMD value (kernel='{args.kernel}') between tensors in '{args.path1}' and '{args.path2}': {result:.6f}")

if __name__ == "__main__":
    main()
