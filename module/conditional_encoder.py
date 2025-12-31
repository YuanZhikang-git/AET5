import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureWiseTransformerEncoder(nn.Module):
    """
    Feature-wise Transformer Encoder for gene expression data.

    Each feature (gene) is treated as a token. The transformer captures
    interactions among features and outputs an embedding for each feature.

    Input shape: [B, num_features]
    Output shape: [B, num_features, feature_embed_dim]
    """

    def __init__(self, num_features, feature_embed_dim=64, nhead=8,
                 num_layers=3, dim_feedforward=256, dropout=0.1):
        """
        Parameters
        ----------
        num_features : int
            Number of features (genes) in input.
        feature_embed_dim : int
            Embedding dimension for each feature.
        nhead : int
            Number of attention heads in the Transformer.
        num_layers : int
            Number of Transformer encoder layers.
        dim_feedforward : int
            Dimension of the feedforward network in Transformer.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        self.num_features = num_features
        self.feature_embed_dim = feature_embed_dim

        # Project scalar feature to embedding
        self.feature_embedding = nn.Linear(1, feature_embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, num_features].

        Returns
        -------
        torch.Tensor
            Feature embeddings of shape [B, num_features, feature_embed_dim].
        """
        x = x.unsqueeze(-1)                     # [B, num_features, 1]
        x = self.feature_embedding(x)           # [B, num_features, feature_embed_dim]
        x = self.transformer(x)                 # [B, num_features, feature_embed_dim]
        return x


class GeneEncoder(nn.Module):
    """
    Gene expression encoder combining:
        1. Optional initial linear layer to expand input features,
        2. Feature-wise Transformer for gene interactions,
        3. Post-transformer MLP projecting each feature embedding to scalar,
        4. Linear projection to latent vector.

    Forward output:
        - latent_vector: [B, latent_size]
        - feature_projection: [B, num_features]
        - optional transformer_output: [B, num_features, hidden_dim] if use_initial_linear=False
    """

    def __init__(self, num_features, input_dim, hidden_dim, latent_size,
                 activation_fn=nn.GELU(), dropout=0.1, nhead=8,
                 num_transformer_layers=2, use_initial_linear=False):
        """
        Parameters
        ----------
        num_features : int
            Number of features after initial linear projection.
        input_dim : int
            Input dimensionality of each sample (number of genes).
        hidden_dim : int
            Feature embedding dimension for Transformer and Post-MLP.
        latent_size : int
            Dimension of the latent representation.
        activation_fn : nn.Module
            Activation function.
        dropout : float
            Dropout probability.
        nhead : int
            Number of attention heads in Transformer.
        num_transformer_layers : int
            Number of Transformer encoder layers.
        use_initial_linear : bool, default False
            If True, bypass initial linear layer and return Transformer embeddings.
        """
        super().__init__()

        self.latent_size = latent_size
        self.num_features = num_features
        self.use_initial_linear = use_initial_linear

        # Initial linear layer (optional)
        if not use_initial_linear:
            self.initial_linear = nn.Sequential(
                nn.Linear(input_dim, num_features),
                activation_fn,
                nn.Dropout(dropout)
            )
        else:
            self.initial_linear = nn.Identity()  # 不做线性投影

        # Feature-wise Transformer
        self.featurewise_transformer = FeatureWiseTransformerEncoder(
            num_features=num_features,
            feature_embed_dim=hidden_dim,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )

        # Post-transformer MLP: project each feature embedding to scalar
        self.post_transformer_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            activation_fn,
            nn.Dropout(dropout)
        )

        # Final latent projection
        self.encoding_to_latent = nn.Linear(num_features, latent_size)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, input_dim].

        Returns
        -------
        latent_vector : torch.Tensor
            Latent representation of shape [B, latent_size].
        transformer_output : torch.Tensor, optional
            Feature-wise Transformer output of shape [B, num_features, hidden_dim],
            returned only if use_initial_linear=True.
        """
        # Optional initial linear projection
        x_input = x if self.use_initial_linear else self.initial_linear(x)  # [B, num_features]

        # Feature-wise Transformer
        transformer_output = self.featurewise_transformer(x_input)           # [B, num_features, hidden_dim]

        # Post-transformer MLP per feature
        feature_projection = self.post_transformer_mlp(transformer_output)   # [B, num_features, 1]
        feature_projection = feature_projection.squeeze(-1)                  # [B, num_features]

        # Latent vector
        latent_vector = self.encoding_to_latent(feature_projection)          # [B, latent_size]

        if self.use_initial_linear:
            return latent_vector, transformer_output
        else:
            return latent_vector


class GeneContrastiveModel(nn.Module):
    """
    Contrastive learning model for gene expression embeddings.

    Takes two augmented views of gene expression and outputs projected
    representations for InfoNCE contrastive loss.
    """

    def __init__(self, encoder, projection_dim=128):
        """
        Parameters
        ----------
        encoder : nn.Module
            Gene encoder that outputs latent representations.
        projection_dim : int
            Dimension of projected embeddings for contrastive loss.
        """
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.latent_size, encoder.latent_size),
            nn.ReLU(),
            nn.Linear(encoder.latent_size, projection_dim)
        )

    def forward(self, x1, x2):
        """
        Forward pass.

        Parameters
        ----------
        x1 : torch.Tensor
            First view tensor [B, input_size].
        x2 : torch.Tensor
            Second view tensor [B, input_size].

        Returns
        -------
        p1, p2 : torch.Tensor
            Projected embeddings for contrastive loss.
        """
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.projector(z1)
        p2 = self.projector(z2)
        return p1, p2


def info_nce_loss(p1, p2, temperature=0.2, bidirectional=True):
    """
    Compute InfoNCE contrastive loss.

    Parameters
    ----------
    p1, p2 : torch.Tensor
        Projected embeddings from two augmented views, shape [B, D].
    temperature : float
        Temperature scaling factor.
    bidirectional : bool
        If True, compute symmetric loss (both directions).

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    B = p1.size(0)
    p1 = F.normalize(p1, dim=1)
    p2 = F.normalize(p2, dim=1)

    logits_12 = torch.matmul(p1, p2.T) / temperature
    labels = torch.arange(B, device=p1.device)
    loss_12 = F.cross_entropy(logits_12, labels)

    if bidirectional:
        logits_21 = torch.matmul(p2, p1.T) / temperature
        loss_21 = F.cross_entropy(logits_21, labels)
        return (loss_12 + loss_21) / 2
    else:
        return loss_12
