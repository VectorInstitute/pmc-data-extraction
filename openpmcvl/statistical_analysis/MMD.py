import argparse

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    From https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html#formal-definition
"""


def load_tensors_to_matrix(file_path):
    """
    Loads a single .pt file containing a matrix.

    Args:
        file_path (str): Path to the .pt file containing the matrix.

    Returns
    -------
        torch.Tensor: The loaded matrix.
    """
    try:
        matrix = torch.load(file_path, weights_only=True)
        if matrix.ndim == 2:
            return matrix
        raise ValueError(f"Loaded tensor is not a 2D matrix: {matrix.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load tensor matrix from {file_path}: {e}")


def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2.0 * XY)


def combine_and_shuffle(A, B):
    """
    Combine two matrices A and B into a single dataset, shuffle, and split.

    Args:
        A (torch.Tensor): First dataset of size (m, D).
        B (torch.Tensor): Second dataset of size (n, D).

    Returns
    -------
        A_prime (torch.Tensor): Shuffled subset of size (m, D) from the combined dataset.
        B_prime (torch.Tensor): Shuffled subset of size (n, D) from the combined dataset.
    """
    # Combine A and B into C
    C = torch.cat((A, B), dim=0)

    # Shuffle C randomly
    indices = torch.randperm(C.size(0))
    C_shuffled = C[indices]

    # Split C into A' and B'
    m, n = A.size(0), B.size(0)
    A_prime = C_shuffled[:m]
    B_prime = C_shuffled[m : m + n]

    return A_prime, B_prime


def compute_p_value(null_distribution, observed_mmd2):
    """
    Computes the p-value as the proportion of null distribution values
    greater than or equal to the observed MMD^2 statistic.

    Args:
        null_distribution (list or array): List of MMD^2 values from the null distribution.
        observed_mmd2 (float): Observed MMD^2 statistic.

    Returns
    -------
        float: The computed p-value.
    """
    count_greater_equal = sum(
        1 for value in null_distribution if value >= observed_mmd2
    )
    p_value = count_greater_equal / len(null_distribution)
    return p_value


def main():
    parser = argparse.ArgumentParser(
        description="Compute MMD between two tensor directories."
    )
    parser.add_argument(
        "path1", type=str, help="Path to the first directory containing .pt files."
    )
    parser.add_argument(
        "path2", type=str, help="Path to the second directory containing .pt files."
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        help="Kernel to use for MMD computation (default: rbf).",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of permuting (default: 10)."
    )
    args = parser.parse_args()

    first_representations = load_tensors_to_matrix(args.path1).to(device)
    second_representations = load_tensors_to_matrix(args.path2).to(device)

    MMD_obs = MMD(first_representations, second_representations, kernel=args.kernel)
    print(f"Observation MMD value: {MMD_obs:.6f}")
    MMD_perms = []

    for i in range(args.n):
        first_representations_prime, second_representations_prime = combine_and_shuffle(
            first_representations, second_representations
        )
        result = MMD(
            first_representations_prime,
            second_representations_prime,
            kernel=args.kernel,
        )
        MMD_perms.append(result)
        print(f"{i}. Permuted MMD value: {result:.6f}")

    p_value = compute_p_value(MMD_perms, MMD_obs)
    print(f"P-value: {p_value}")


if __name__ == "__main__":
    main()
