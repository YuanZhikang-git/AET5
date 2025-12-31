import os

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.torch_dataset import ExpressionDataset
from module.conditional_encoder import GeneEncoder, GeneContrastiveModel, info_nce_loss
from utiles.data_augmentation import load_or_compute_gene_statistics, CombinedAugmentation


def train(
        data_path,
        statistics_path,
        base_std=0.02,
        dropout_rate=0.2,
        scaling_range=(0.85, 1.15),
        p_noise=0,
        p_structured_noise=0,
        p_dropout=1,
        p_scaling=0,
        max_augments=2,
        batch_size=256,
        input_dim=978,
        num_features=320,
        num_transformer_layers=2,
        hidden_dim=64,
        latent_size=128,
        projection_dim=128,
        lr=1e-4,
        epochs=100,
        temperature=0.2,
        save_dir="./checkpoints"
):
    """
    Train a gene contrastive encoder inference InfoNCE loss with data augmentations.

    Parameters
    ----------
    data_path : str
        Path to CSV file containing gene expression data.
    statistics_path : str
        Directory to save/load precomputed gene statistics (std and covariance).
    base_std : float
        Base standard deviation for adaptive Gaussian noise.
    dropout_rate : float
        Dropout probability for gene-wise dropout augmentation.
    scaling_range : tuple of float
        Range of multiplicative scaling factors for scaling augmentation.
    p_noise : float
        Probability of applying adaptive Gaussian noise.
    p_structured_noise : float
        Probability of applying structured Gaussian noise.
    p_dropout : float
        Probability of applying gene-wise dropout.
    p_scaling : float
        Probability of applying scaling augmentation.
    max_augments : int
        Maximum number of augmentations applied per batch.
    batch_size : int
        Batch size for DataLoader.
    input_dim : int
        Input dimension of gene expression vectors.
    num_features : int
        Number of features after optional initial linear projection.
    num_transformer_layers : int
        Number of transformer encoder layers.
    hidden_dim : int
        Feature embedding dimension in Transformer.
    latent_size : int
        Dimension of latent vector output from encoder.
    projection_dim : int
        Dimension of projection head for contrastive loss.
    lr : float
        Learning rate for Adam optimizer.
    epochs : int
        Number of training epochs.
    temperature : float
        Temperature parameter for InfoNCE loss.
    save_dir : str
        Directory to save trained encoder and loss history.

    Returns
    -------
    None
        Saves encoder weights and training loss CSV to `save_dir`.
    """

    # Create directory for saving outputs
    os.makedirs(save_dir, exist_ok=True)

    # Load or compute gene-wise statistics for augmentations
    gene_std, gene_cov = load_or_compute_gene_statistics(data_path, statistics_path)

    # Initialize combined augmentation module
    combined_augmentation = CombinedAugmentation(
        gene_std=gene_std,
        cov_matrix=gene_cov,
        base_std=base_std,
        dropout_rate=dropout_rate,
        scaling_range=scaling_range,
        p_noise=p_noise,
        p_structured_noise=p_structured_noise,
        p_dropout=p_dropout,
        p_scaling=p_scaling,
        max_augments=max_augments
    )

    # Build dataset and DataLoader
    dataset = ExpressionDataset(data_path)

    def collate_fn(batch):
        """
        Collate function generating two augmented views per batch.

        Parameters
        ----------
        batch : list of tuples
            Each element is (sample_id, expression_tensor)

        Returns
        -------
        x1, x2 : torch.Tensor
            Two augmented views of shape [B, num_genes]
        """
        _, _, expressions = zip(*batch)
        expressions = torch.stack(expressions)  # [B, num_genes]

        # Generate two random augmented views
        x1 = combined_augmentation(expressions)
        x2 = combined_augmentation(expressions)
        return x1, x2

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Initialize encoder and contrastive model
    encoder = GeneEncoder(
        input_dim=input_dim,
        num_features=num_features,
        num_transformer_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        latent_size=latent_size
    )
    model = GeneContrastiveModel(encoder, projection_dim=projection_dim)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    loss_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x1, x2 in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x1 = x1.to(device)
            x2 = x2.to(device)

            optimizer.zero_grad()
            p1, p2 = model(x1, x2)
            loss = info_nce_loss(p1, p2, temperature=temperature)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x1.size(0)

        avg_loss = total_loss / len(dataset)
        loss_history.append({"epoch": epoch + 1, "avg_loss": avg_loss})
        print(f"Epoch {epoch + 1}, Avg InfoNCE Loss: {avg_loss:.4f}")

    # Save trained encoder only
    torch.save(encoder.state_dict(), os.path.join(save_dir, "conditional_encoder.pt"))

    # Save loss history to CSV
    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(os.path.join(save_dir, "conditional_encoder_loss.csv"), index=False)

