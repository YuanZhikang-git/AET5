import torch
import numpy as np
import csv
from rdkit import Chem, RDLogger
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from module.molecular_generator import ConditionalMolT5

def generate_smiles_from_features(
        feature_csv,
        cond_model_pt,
        t5_path,
        output_csv,
        cond_tokens=32,
        cond_mode="vector",
        num_augment=99,
        noise_std=0.1,
        num_smiles_per_sample=5
):
    """
    Generate SMILES from feature vectors with Gaussian noise augmentation.

    Each input feature vector produces 1 original + num_augment noisy samples,
    each sample generates num_smiles_per_sample SMILES. Only valid and unique
    SMILES are kept, all SMILES for the same original sample are saved in one row.

    Args:
        feature_csv (str): Path to input CSV (no header, each row is a feature vector).
        cond_model_pt (str): Path to saved ConditionalMolT5 .pt weights.
        t5_path (str): Path to pretrained T5 model used for ConditionalMolT5.
        output_csv (str): Path to save generated SMILES CSV.
        cond_dim (int): Feature vector dimension (must match training).
        cond_tokens (int): Number of virtual condition tokens.
        cond_mode (str): Conditioning mode ("vector" or "token_emb").
        num_augment (int): Number of augmented samples per input.
        noise_std (float): Std of Gaussian noise for augmentation (0-0.1).
        num_smiles_per_sample (int): Number of SMILES to generate per sample.
    """
    RDLogger.DisableLog('rdApp.*')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize tokenizer and base T5 model
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_path)

    # Initialize ConditionalMolT5
    cond_model = ConditionalMolT5(
        model=t5_model,
        tokenizer=tokenizer,
        cond_tokens=cond_tokens,
        cond_mode=cond_mode
    )
    cond_model.load_state_dict(torch.load(cond_model_pt, map_location=device))
    cond_model.to(device)
    cond_model.eval()

    # Load features
    features = np.loadtxt(feature_csv, delimiter=',', dtype=np.float32)

    with open(output_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)

        for i in tqdm(range(features.shape[0]), desc="Generating SMILES"):
            orig_vec = features[i]

            # Original + augmented vectors
            aug_vectors = [orig_vec] + [
                orig_vec + np.random.normal(0, noise_std, size=orig_vec.shape)
                for _ in range(num_augment)
            ]

            all_generated_smiles = set()

            for vec in aug_vectors:
                vec_tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    gen_smiles_list = cond_model.generate_smiles(
                        cond_input=vec_tensor,
                        num_return_sequences=num_smiles_per_sample,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95
                    )

                # Filter valid and unique SMILES
                for smi in gen_smiles_list:
                    if smi not in all_generated_smiles:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            all_generated_smiles.add(smi)

            # Write all unique valid SMILES for this sample in one row
            writer.writerow(list(all_generated_smiles))

    print(f"SMILES generation complete. Results saved to {output_csv}")
