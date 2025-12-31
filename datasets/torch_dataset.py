import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset
import pandas as pd


class ExpressionDataset(Dataset):
    """
    PyTorch Dataset for transcriptomic expression data.

    Each sample consists of:
        - sample ID
        - SMILES string
        - gene expression vector (all columns from the third column onward)

    The input CSV file is expected to have no header:
        column 0 : sample ID
        column 1 : SMILES string
        column 2+ : gene expression values
    """

    def __init__(self, csv_path):
        """
        Parameters
        ----------
        csv_path : str
            Path to the CSV file without header.
        """
        self.df = pd.read_csv(csv_path, header=None)

        if self.df.shape[1] < 3:
            raise ValueError(
                f"Expected at least 3 columns (ID, SMILES, expressions), got {self.df.shape[1]}"
            )

        # Cache sample IDs and SMILES
        self.sample_ids = self.df.iloc[:, 0].astype(str).values
        self.smiles = self.df.iloc[:, 1].astype(str).values
        # Expression values from the 3rd column onward
        self.expression = self.df.iloc[:, 2:].values.astype("float32")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """
        Returns
        -------
        sample_id : str
            Sample identifier.
        smiles : str
            SMILES string corresponding to the sample.
        expression : torch.FloatTensor
            Gene expression vector (variable length).
        """
        sample_id = self.sample_ids[idx]
        smiles = self.smiles[idx]
        expression = torch.from_numpy(self.expression[idx])

        return sample_id, smiles, expression


class SmilesDataset(Dataset):
    """
    PyTorch Dataset for paired gene expression and SMILES data.

    Each sample consists of:
        - sample ID (str)
        - tokenized SMILES sequence (list of int)
        - gene expression vector (np.ndarray of float32)

    The input CSV file is expected to have no header:
        column 0 : sample ID
        column 1 : SMILES string
        column 2+ : gene expression values
    """

    def __init__(self, csv_path, tokenizer, variant=True):
        """
        Initialize the dataset.

        Parameters
        ----------
        csv_path : str
            Full path to the CSV file containing the data.
        tokenizer : object
            Tokenizer used for SMILES sequences (e.g., HuggingFace T5Tokenizer).
        variant : bool, default True
            If True, apply randomized (non-canonical) SMILES augmentation.
        """
        self.data = pd.read_csv(csv_path, header=None).dropna(how='any')
        self.tokenizer = tokenizer
        self.variant = variant

        if hasattr(tokenizer, "pad_token_id"):
            self.pad_token_id = tokenizer.pad_token_id
        else:
            self.pad_token_id = None

        if self.variant:
            self.data[1] = self.data[1].apply(self._randomize_smiles)

    def _randomize_smiles(self, smi):
        """
        Generate a randomized (non-canonical) SMILES string.

        Parameters
        ----------
        smi : str
            Original canonical SMILES string.

        Returns
        -------
        str
            Randomized SMILES string representing the same molecule.
        """
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True)

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        sample_id : str
            Identifier of the sample.
        encoded_smi : list[int]
            Tokenized SMILES sequence.
        gene_expr : np.ndarray
            Gene expression vector (dtype float32).
        """
        sample_id = str(self.data.iloc[idx, 0])
        smi = self.data.iloc[idx, 1]

        if hasattr(self.tokenizer, "encode"):
            encoded_smi = self.tokenizer.encode(smi, add_special_tokens=True)
        else:
            encoded_smi = self.tokenizer(smi)

        gene_expr = self.data.iloc[idx, 2:].values.astype("float32")

        return sample_id, encoded_smi, gene_expr