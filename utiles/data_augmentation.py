import os

import numpy as np
import pandas as pd
import torch
import random

def load_or_compute_gene_statistics(csv_file, output_dir):
    """
    Load or compute gene-wise statistics for augmentations.

    Parameters
    ----------
    csv_file : str
        Path to CSV file with gene expression data. Assumes no header,
        first column is ID, second is SMILES, remaining columns are gene expression.
    output_dir : str
        Directory to save or load precomputed statistics.

    Returns
    -------
    gene_std : torch.Tensor
        Gene-wise standard deviation, shape [num_genes].
    gene_cov : torch.Tensor
        Gene-gene covariance matrix, shape [num_genes, num_genes].
    """
    os.makedirs(output_dir, exist_ok=True)
    std_path = os.path.join(output_dir, "gene_std.pt")
    cov_path = os.path.join(output_dir, "gene_cov.pt")

    # 如果文件存在，则直接加载
    if os.path.exists(std_path) and os.path.exists(cov_path):
        gene_std = torch.load(std_path, weights_only=True)
        gene_cov = torch.load(cov_path, weights_only=True)
        return gene_std, gene_cov

    # 读取 CSV
    df = pd.read_csv(csv_file, header=None)
    expr = df.iloc[:, 2:].values.astype(np.float32)  # [num_samples, num_genes]
    expr_tensor = torch.from_numpy(expr)

    # 计算基因标准差
    gene_std = expr_tensor.std(dim=0)  # [num_genes]

    # 计算基因协方差
    expr_centered = expr_tensor - expr_tensor.mean(dim=0, keepdim=True)
    gene_cov = (expr_centered.T @ expr_centered) / (expr_tensor.size(0) - 1)  # [num_genes, num_genes]

    # 保存
    torch.save(gene_std, std_path)
    torch.save(gene_cov, cov_path)

    return gene_std, gene_cov

class AdaptiveGaussianNoise:
    """
    Adaptive gene-wise Gaussian noise augmentation.

    This module injects zero-mean Gaussian noise into gene expression vectors,
    where the noise magnitude is adaptively scaled according to the relative
    variability of each gene across the dataset.

    Specifically, the standard deviation of the noise for gene i is defined as:
        σ_i = base_std · (std_i / max(std))

    This design preserves relative gene-wise variability while avoiding
    excessive perturbation caused by absolute variance differences.
    """

    def __init__(self, gene_std, base_std=0.02, epsilon=1e-8):
        """
        Parameters
        ----------
        gene_std : torch.Tensor
            Empirical standard deviation of each gene across the dataset,
            shape (num_genes,).
        base_std : float, optional
            Global scaling factor controlling overall noise intensity.
        epsilon : float, optional
            Small constant to avoid division by zero.
        """
        max_val = torch.max(gene_std) if gene_std.numel() > 0 else 1.0
        # Normalize gene-wise variability to [0, 1]
        self.noise_scale = gene_std / (max_val + epsilon)
        self.base_std = base_std
        self.epsilon = epsilon

    def __call__(self, x):
        """
        Apply adaptive Gaussian noise.

        Parameters
        ----------
        x : torch.Tensor
            Input gene expression tensor of shape (B, num_genes).

        Returns
        -------
        torch.Tensor
            Noise-perturbed gene expression tensor.
        """
        noise_scale = self.noise_scale.to(device=x.device, dtype=x.dtype)
        noise_std = self.base_std * noise_scale
        noise = torch.randn_like(x) * noise_std
        return x + noise


class StructuredGaussianNoise:
    """
    Structured Gaussian noise augmentation with gene–gene correlations.

    Noise is sampled from a multivariate Gaussian distribution whose covariance
    matrix is derived from the empirical gene–gene covariance, enabling
    biologically correlated perturbations across genes.
    """

    def __init__(self, cov_matrix, base_std=0.02, epsilon=1e-6):
        """
        Parameters
        ----------
        cov_matrix : torch.Tensor
            Empirical gene–gene covariance matrix, shape (num_genes, num_genes).
        base_std : float, optional
            Global scaling factor controlling overall noise intensity.
        epsilon : float, optional
            Diagonal regularization term for numerical stability.
        """
        self.base_std = base_std
        self.epsilon = epsilon

        # Normalize covariance magnitude to control perturbation scale
        max_val = torch.max(torch.abs(cov_matrix)) if cov_matrix.numel() > 0 else 1.0
        scaling_factor = (base_std ** 2) / (max_val + epsilon)
        self.cov_matrix = cov_matrix * scaling_factor

        self.gene_dim = cov_matrix.size(0)
        self._precompute_distribution()

    def _precompute_distribution(self):
        """
        Precompute the Cholesky decomposition of the regularized covariance
        matrix for efficient sampling.
        """
        cov_reg = self.cov_matrix + torch.eye(self.gene_dim) * self.epsilon
        try:
            self.cholesky = torch.linalg.cholesky(cov_reg)
        except RuntimeError:
            # Fallback to diagonal approximation if covariance is not PSD
            diag = torch.diag(self.cov_matrix).clamp(min=self.epsilon)
            self.cholesky = torch.diag_embed(torch.sqrt(diag))

    def __call__(self, x):
        """
        Apply structured Gaussian noise.

        Parameters
        ----------
        x : torch.Tensor
            Input gene expression tensor of shape (B, num_genes).

        Returns
        -------
        torch.Tensor
            Noise-perturbed gene expression tensor with preserved
            gene–gene correlation structure.
        """
        B = x.size(0)
        device = x.device

        base_noise = torch.randn(B, self.gene_dim, device=device)
        structured_noise = base_noise @ self.cholesky.to(device)

        return x + structured_noise


def random_dropout(x, dropout_rate=0.2):
    """
    Gene-wise adaptive dropout augmentation.

    Genes with higher variance across the batch are assigned higher dropout
    probabilities, encouraging robustness to partial expression perturbations.
    """
    gene_variance = x.var(dim=0)
    dropout_prob = (gene_variance / gene_variance.max()).clamp(
        min=0.1, max=dropout_rate
    )

    mask = torch.rand_like(x) > dropout_prob
    return x * mask


def fourier_perturbation(x, perturbation_strength=0.1):
    """
    Frequency-domain perturbation of gene expression signals.

    Applies mild Gaussian noise to both real and imaginary components in the
    Fourier domain, with perturbation strength modulated by gene-wise variance.
    """
    signal_std = x.std(dim=0)
    perturbation_factor = perturbation_strength * (signal_std / signal_std.max())

    x_fft = torch.fft.fft(x)
    x_fft_real = x_fft.real + torch.randn_like(x_fft.real) * perturbation_factor
    x_fft_imag = x_fft.imag + torch.randn_like(x_fft.imag) * perturbation_factor

    x_perturbed = torch.fft.ifft(
        torch.complex(x_fft_real, x_fft_imag)
    ).real
    return x_perturbed


def scaling_transformation(x, scaling_range=(0.85, 1.15)):
    """
    Multiplicative scaling augmentation.

    Expression values are scaled by random factors, further modulated by
    gene-wise variability to preserve relative expression patterns.
    """
    scaling_factors = (
        torch.rand_like(x) * (scaling_range[1] - scaling_range[0])
        + scaling_range[0]
    )

    gene_std = x.std(dim=0)
    scaling_factors *= (gene_std / gene_std.max()).clamp(min=0.8, max=1.2)

    return x * scaling_factors


class CombinedAugmentation:
    """
    Stochastic combination of multiple gene expression augmentations.

    At each call, a subset of augmentations is randomly selected and applied
    sequentially, including adaptive Gaussian noise, structured Gaussian noise,
    dropout, and scaling transformations.
    """

    def __init__(
        self,
        gene_std,
        cov_matrix=None,
        base_std=0.02,
        dropout_rate=0.2,
        scaling_range=(0.85, 1.15),
        p_noise=0.6,
        p_structured_noise=0.4,
        p_dropout=0.6,
        p_scaling=0.6,
        max_augments=2
    ):
        """
        Parameters
        ----------
        gene_std : torch.Tensor
            Gene-wise empirical standard deviation.
        cov_matrix : torch.Tensor or None
            Gene–gene covariance matrix for structured noise.
        p_* : float
            Probabilities controlling whether each augmentation is applied.
        max_augments : int
            Maximum number of augmentations applied sequentially.
        """
        self.noise_aug = AdaptiveGaussianNoise(gene_std, base_std)

        self.use_structured_noise = cov_matrix is not None
        if self.use_structured_noise:
            self.structured_noise_aug = StructuredGaussianNoise(
                cov_matrix=cov_matrix,
                base_std=base_std
            )

        self.dropout_rate = dropout_rate
        self.scaling_range = scaling_range

        self.p_noise = p_noise
        self.p_structured_noise = p_structured_noise
        self.p_dropout = p_dropout
        self.p_scaling = p_scaling

        self.max_augments = max_augments

    def __call__(self, x, epoch=None):
        """
        Apply a random combination of augmentations.

        Parameters
        ----------
        x : torch.Tensor
            Input gene expression tensor.
        epoch : int or None
            Reserved for future scheduling strategies.

        Returns
        -------
        torch.Tensor
            Augmented gene expression tensor.
        """
        candidates = []

        if random.random() < self.p_noise:
            candidates.append(lambda x: self.noise_aug(x))

        if self.use_structured_noise and random.random() < self.p_structured_noise:
            candidates.append(lambda x: self.structured_noise_aug(x))

        if random.random() < self.p_dropout:
            candidates.append(lambda x: random_dropout(x, self.dropout_rate))

        if random.random() < self.p_scaling:
            candidates.append(lambda x: scaling_transformation(x, self.scaling_range))

        # Ensure at least one augmentation is applied
        if len(candidates) == 0:
            candidates = [random.choice([
                lambda x: self.noise_aug(x),
                lambda x: random_dropout(x, self.dropout_rate),
                lambda x: scaling_transformation(x, self.scaling_range)
            ])]

        # Limit the number of applied augmentations
        if len(candidates) > self.max_augments:
            candidates = random.sample(candidates, self.max_augments)

        random.shuffle(candidates)

        for aug in candidates:
            x = aug(x)

        return x
