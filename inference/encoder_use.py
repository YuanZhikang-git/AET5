import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.torch_dataset import ExpressionDataset
from module.conditional_encoder import GeneEncoder


def embed_samples(data_path, encoder_path, output_csv,
                  input_dim=978, num_features=320, hidden_dim=64, latent_size=128,
                  batch_size=256):
    """
    Embed all samples inference a trained gene encoder and save latent vectors to CSV.

    Parameters
    ----------
    data_path : str
        Path to CSV file containing the expression data.
    encoder_path : str
        Path to trained encoder state dict (.pt file).
    output_csv : str
        Path to save the resulting CSV with embeddings.
    input_dim : int
        Input dimension of gene expression vectors.
    num_features : int
        Number of features in encoder.
    hidden_dim : int
        Hidden dimension in encoder.
    latent_size : int
        Dimension of the latent embedding.
    batch_size : int
        Batch size for DataLoader.

    Returns
    -------
    None
        Saves CSV with columns: ID, SMILES, latent_0, ..., latent_{latent_size-1}.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = ExpressionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize encoder
    encoder = GeneEncoder(
        input_dim=input_dim,
        num_features=num_features,
        hidden_dim=hidden_dim,
        latent_size=latent_size
    )
    encoder.load_state_dict(torch.load(encoder_path, map_location=device,weights_only=False))
    encoder = encoder.to(device)
    encoder.eval()

    all_ids = []
    all_smiles = []
    all_latents = []

    with torch.no_grad():
        for sample_ids, smiles, expressions in tqdm(dataloader, desc="Embedding samples"):
            expressions = expressions.to(device)
            latent_vectors = encoder(expressions)  # [B, latent_size]

            all_ids.extend(sample_ids)
            all_smiles.extend(smiles)
            all_latents.append(latent_vectors.cpu())

    # Concatenate all latent vectors
    all_latents = torch.cat(all_latents, dim=0).numpy()  # [num_samples, latent_size]

    # Build DataFrame
    columns = ["ID", "SMILES"] + [f"latent_{i}" for i in range(latent_size)]
    df = pd.DataFrame(
        data=np.hstack([np.array(all_ids).reshape(-1, 1),
                        np.array(all_smiles).reshape(-1, 1),
                        all_latents]),
        columns=columns
    )

    # Save CSV without header
    df.to_csv(output_csv, index=False, header=False)
    print(f"Embeddings saved to {output_csv}")