import argparse

import numpy as np
import torch
from sklearn.decomposition import PCA


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


def MMD(x, y, kernel, bandwidth):
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
        XX += bandwidth**2 * (bandwidth**2 + dxx) ** -1
        YY += bandwidth**2 * (bandwidth**2 + dyy) ** -1
        XY += bandwidth**2 * (bandwidth**2 + dxy) ** -1

    if kernel == "rbf":  # Radial Basis Function (RBF) kernel
        XX += torch.exp(-0.5 * dxx / bandwidth)
        YY += torch.exp(-0.5 * dyy / bandwidth)
        XY += torch.exp(-0.5 * dxy / bandwidth)

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


def create_subsamples_from_one_matrix(matrix, subsample_size, percentage):
    """
    Creates two subsamples from a matrix with a specified percentage of similar data points
    and converts them to PyTorch tensors.

    Args:
        matrix (np.ndarray): Input matrix where each row is a data point.
        subsample_size (int): Number of data points in each subsample.
        percentage (float): Percentage (0-100) of similar data points between the two subsamples.

    Returns
    -------
        tuple: Two subsamples as PyTorch tensors.
    """
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100.")
    if subsample_size > len(matrix):
        raise ValueError(
            "Subsample size cannot be larger than the number of data points in the matrix."
        )

    np.random.shuffle(matrix)

    num_shared_points = int((percentage / 100) * subsample_size)

    num_unique_points = subsample_size - num_shared_points

    shared_indices = np.random.choice(len(matrix), num_shared_points, replace=False)
    shared_points = matrix[shared_indices]

    remaining_indices = list(set(range(len(matrix))) - set(shared_indices))
    unique_indices_1 = np.random.choice(
        remaining_indices, num_unique_points, replace=False
    )
    unique_points_1 = matrix[unique_indices_1]

    remaining_indices = list(set(remaining_indices) - set(unique_indices_1))
    unique_indices_2 = np.random.choice(
        remaining_indices, num_unique_points, replace=False
    )
    unique_points_2 = matrix[unique_indices_2]

    subsample_1 = np.vstack((shared_points, unique_points_1))
    subsample_2 = np.vstack((shared_points, unique_points_2))

    tensor_1 = torch.tensor(subsample_1, dtype=torch.float32)
    tensor_2 = torch.tensor(subsample_2, dtype=torch.float32)

    return tensor_1.to(device), tensor_2.to(device)


def create_subsamples_from_two_matrices(matrix1, matrix2, subsample_size, percentage):
    """
    Creates two subsamples from two given matrices with a specified percentage of similar data points
    and converts them to PyTorch tensors.

    Args:
        matrix1 (np.ndarray): First input matrix where each row is a data point.
        matrix2 (np.ndarray): Second input matrix where each row is a data point.
        subsample_size (int): Number of data points in each subsample.
        percentage (float): Percentage (0-100) of similar data points between the two subsamples.

    Returns
    -------
        tuple: Two subsamples as PyTorch tensors.
    """
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100.")
    if subsample_size > len(matrix1) or subsample_size > len(matrix2):
        raise ValueError(
            "Subsample size cannot be larger than the number of data points in either matrix."
        )
    np.random.shuffle(matrix1)
    np.random.shuffle(matrix2)

    num_shared_points = int((percentage / 100) * subsample_size)

    num_unique_points = subsample_size - num_shared_points

    shared_indices_matrix1 = np.random.choice(
        len(matrix1), num_shared_points, replace=False
    )
    shared_points_matrix1 = matrix1[shared_indices_matrix1]

    shared_indices_matrix2 = np.random.choice(
        len(matrix2), num_shared_points, replace=False
    )
    shared_points_matrix2 = matrix2[shared_indices_matrix2]

    remaining_indices_matrix1 = list(
        set(range(len(matrix1))) - set(shared_indices_matrix1)
    )
    unique_indices_matrix1 = np.random.choice(
        remaining_indices_matrix1, num_unique_points, replace=False
    )
    unique_points_matrix1 = matrix1[unique_indices_matrix1]

    remaining_indices_matrix2 = list(
        set(range(len(matrix2))) - set(shared_indices_matrix2)
    )
    unique_indices_matrix2 = np.random.choice(
        remaining_indices_matrix2, num_unique_points, replace=False
    )
    unique_points_matrix2 = matrix2[unique_indices_matrix2]
    
    subsample1 = np.vstack((shared_points_matrix1, unique_points_matrix1))
    subsample2 = np.vstack((shared_points_matrix2, unique_points_matrix2))

    tensor1 = torch.tensor(subsample1, dtype=torch.float32)
    tensor2 = torch.tensor(subsample2, dtype=torch.float32)

    return tensor1.to(device), tensor2.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Compute MMD between two tensor directories."
    )
    parser.add_argument(
        "path1",
        type=str,
        help="Path to the first directory containing .pt files.",
    )
    parser.add_argument(
        "--path2", type=str, help="Path to the second directory containing .pt files."
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        help="Kernel to use for MMD computation (default: rbf).",
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of permuting (default: 10)."
    )
    parser.add_argument(
        "--bandwidth",
        type=int,
        default=10,
        help="Number of bandwidth for kernel (default: 10).",
    )
    parser.add_argument(
        "--sampling_type",
        type=str,
        choices=["one_matrix", "two_matrices"],
        help="Type of sampling: 'one_matrix' for a single matrix or 'two_matrices' for two matrices.",
    )
    parser.add_argument(
        "--subsample_size",
        type=int,
        default=10,
        help="Number of data points in each subsample (default: 10).",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=50.0,
        help="Percentage of similar data points between subsamples (default: 50).",
    )
    args = parser.parse_args()

    first_representations = load_tensors_to_matrix(args.path1)

    if args.sampling_type == "one_matrix":
        first_representations, second_representations = create_subsamples_from_one_matrix(
            first_representations, args.subsample_size, args.percentage
        )
    elif args.sampling_type == "two_matrices":
        second_representations = load_tensors_to_matrix(args.path2)
        first_representations, second_representations = (
            create_subsamples_from_two_matrices(
                first_representations,
                second_representations,
                args.subsample_size,
                args.percentage,
            )
        )
        if first_representations.shape[1] != second_representations.shape[1]:
            if first_representations.shape[1] > second_representations.shape[1]:
                pca = PCA(n_components=second_representations.shape[1])
                first_representations_reduced = pca.fit_transform(first_representations.cpu().numpy())
                first_representations = torch.tensor(first_representations_reduced, dtype=torch.float32, device=first_representations.device)
                pca = PCA(n_components=second_representations.shape[1])
                second_representations_reduced = pca.fit_transform(second_representations.cpu().numpy())
                second_representations = torch.tensor(second_representations_reduced, dtype=torch.float32, device=second_representations.device)
            else:
                pca = PCA(n_components=first_representations.shape[1])
                second_representations_reduced = pca.fit_transform(second_representations.cpu().numpy())
                second_representations = torch.tensor(second_representations_reduced, dtype=torch.float32, device=second_representations.device)
                pca = PCA(n_components=first_representations.shape[1])
                first_representations_reduced = pca.fit_transform(first_representations.cpu().numpy())
                first_representations = torch.tensor(first_representations_reduced, dtype=torch.float32, device=first_representations.device)
    elif args.sampling_type is None:
        first_representations = first_representations.to(device)
        second_representations = load_tensors_to_matrix(args.path2).to(device)

    MMD_obs = MMD(
        first_representations,
        second_representations,
        kernel=args.kernel,
        bandwidth=args.bandwidth,
    )
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
            bandwidth=args.bandwidth,
        )
        MMD_perms.append(result)
        print(f"{i}. Permuted MMD value: {result:.6f}")

    p_value = compute_p_value(MMD_perms, MMD_obs)
    print(f"P-value: {p_value} \n")

    MMD_perms = torch.stack(MMD_perms)
    overall_min = MMD_perms.min()
    overall_max = MMD_perms.max()

    print(f"Observation MMD value: {MMD_obs} \n")

    print(f"Overall Min: {overall_min}")
    print(f"Overall Max: {overall_max}")

    # Calculate the 95% percentile range
    lower_percentile = torch.quantile(MMD_perms, 0.025)  # 2.5% percentile
    upper_percentile = torch.quantile(MMD_perms, 0.975)  # 97.5% percentile

    # Filter values within the 95% range
    middle_values = MMD_perms[
        (MMD_perms >= lower_percentile) & (MMD_perms <= upper_percentile)
    ]

    # Find the min and max of the middle 95%
    middle_min = middle_values.min()
    middle_max = middle_values.max()
    
    
    torch.set_printoptions(precision=10)
    print(MMD_perms)

    print(f"Middle 95% Min: {middle_min}")
    print(f"Middle 95% Max: {middle_max}")


if __name__ == "__main__":
    main()
