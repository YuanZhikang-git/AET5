import csv
import os
import warnings

import torch
from rdkit import RDLogger
from torch.utils.data import DataLoader, random_split
from torch import optim
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import random

from datasets.torch_dataset import SmilesDataset
from module.molecular_generator import ConditionalMolT5
from utiles.gen_evl_tool import is_valid_smiles, mol_to_fp, max_similarity, compute_diversity


def train_conditional_molt5(
    csv_path,
    t5_path="./MolT5-small-caption2smiles",
    train_batch_size=32,
    test_batch_size=16,
    cond_tokens=32,
    cond_mode="vector",
    lr=1e-4,
    epochs=200,
    save_dir="./checkpoints"
):
    """
    Train ConditionalMolT5 with progress bars. Each epoch evaluates
    the model on the validation set and saves metrics to CSV and model weights.

    Args:
        csv_path (str): CSV file (no header, first col=ID, second col=SMILES, rest=gene vectors).
        t5_path (str): Path to pretrained MolT5.
        train_batch_size (int): Training batch size.
        test_batch_size (int): Validation batch size.
        cond_tokens (int): Number of virtual condition tokens.
        cond_mode (str): Conditioning mode ("vector" or "token_emb").
        lr (float): Learning rate.
        epochs (int): Number of training epochs.
        save_dir (str): Directory to save model weights and metrics CSV.

    Returns:
        cond_model: Trained ConditionalMolT5.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    dataset = SmilesDataset(csv_path=csv_path, tokenizer=tokenizer, variant=True)

    # 9:1 train/val split
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    def collate_fn(batch):
        ids, smiles_list, gene_list = zip(*batch)
        smi_tensors = [torch.tensor(smi, dtype=torch.long) for smi in smiles_list]
        smi_tensors = torch.nn.utils.rnn.pad_sequence(
            smi_tensors, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = (smi_tensors != tokenizer.pad_token_id).long()
        gene_tensors = torch.tensor(np.stack(gene_list), dtype=torch.float32)
        return ids, smi_tensors, gene_tensors, attention_mask

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize models
    molt5_model = T5ForConditionalGeneration.from_pretrained(t5_path)
    cond_model = ConditionalMolT5(
        model=molt5_model,
        tokenizer=tokenizer,
        cond_dim=dataset.data.shape[1]-2,
        cond_tokens=cond_tokens,
        cond_mode=cond_mode
    )
    cond_model.to(device)

    optimizer = optim.Adam(cond_model.parameters(), lr=lr)

    # Prepare save directory and metrics CSV
    os.makedirs(save_dir, exist_ok=True)
    metrics_csv = os.path.join(save_dir, "generator_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss", "validity", "uniqueness", "novelty", "diversity"])

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        cond_model.train()
        epoch_losses = []
        for batch_idx, (ids, smi_tensors, gene_tensors, attention_mask) in enumerate(
            tqdm(train_loader, desc="Training Batches", ncols=100)
        ):
            smi_tensors = smi_tensors.to(device)
            gene_tensors = gene_tensors.to(device)
            attention_mask = attention_mask.to(device)

            labels = smi_tensors.clone()
            labels[labels == tokenizer.pad_token_id] = -100  # ignore padding

            optimizer.zero_grad()
            outputs = cond_model(
                cond_input=gene_tensors,
                decoder_input_ids=smi_tensors,
                decoder_attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)

        # Evaluate
        real_smiles, gen_smiles, metrics = evaluate_conditional_model(
            val_loader, cond_model, tokenizer, device=device
        )

        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                avg_loss,
                metrics["validity"],
                metrics["uniqueness"],
                metrics["novelty"],
                metrics["diversity"]
            ])

        print(
            f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | "
            f"Validity: {metrics['validity']:.2f}% | "
            f"Uniqueness: {metrics['uniqueness']:.2f}% | "
            f"Novelty: {metrics['novelty']:.2f}% | "
            f"Diversity: {metrics['diversity']:.4f}"
        )

    # Save only ConditionalMolT5 weights
    torch.save(cond_model.state_dict(), os.path.join(save_dir, "cond_model.pt"))

    print(f"ConditionalMolT5 weights saved to {save_dir}")

    return cond_model, train_loader, val_loader, tokenizer


def evaluate_conditional_model(
    data_loader,
    model,
    tokenizer,
    device,
    start_token_ids=[3, 205],
    diversity_threshold=0.7,
    random_state=42
):
    """
    Evaluate ConditionalMolT5 on dataset:
        - Generate SMILES
        - Apply selection rules (valid, unique, diverse)
        - Compute Validity, Uniqueness, Novelty, Diversity

    Args:
        data_loader : DataLoader
        model       : ConditionalMolT5
        tokenizer   : T5Tokenizer
        device      : 'cuda' or 'cpu'
        start_token_ids : list of start token ids for decoder
        diversity_threshold : float, max Tanimoto similarity for diverse selection
        random_state : int

    Returns:
        real_smiles_all : list[str]
        gen_smiles_all  : list[str]
        metrics        : dict of validity, uniqueness, novelty, diversity
    """
    RDLogger.DisableLog('rdApp.*')
    warnings.filterwarnings("ignore")
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    model.eval()
    model.to(device)

    gen_smiles_all = []
    real_smiles_all = []
    seen = set()
    existing_fps = []

    with torch.no_grad():
        for ids, smi_tensors, gene_tensors, _ in tqdm(data_loader, desc="Generating SMILES"):
            gene_tensors = gene_tensors.to(device)

            # Decode real SMILES
            real_smiles_batch = [
                tokenizer.decode(smi_tensors[i].tolist(), skip_special_tokens=True).strip()
                for i in range(smi_tensors.size(0))
            ]

            # Generate candidates for each start token
            batch_candidates = []
            for start_id in start_token_ids:
                generated_ids = model.generate(
                    cond_input=gene_tensors,
                    decoder_start_token_id=start_id,
                    max_length=200,
                    do_sample=True,
                    top_k=100,
                    top_p=0.95,
                    temperature=1.1,
                )
                decoded = [tokenizer.decode(ids, skip_special_tokens=True).strip()
                           for ids in generated_ids]
                batch_candidates.append(decoded)

            # Select final SMILES
            for i in range(len(gene_tensors)):
                real_smiles_all.append(real_smiles_batch[i])
                candidates = [batch_candidates[j][i] for j in range(len(start_token_ids))]
                mol_candidates = [(s, is_valid_smiles(s)) for s in candidates]

                final_smi = None

                # valid + unique + diverse
                for s, m in mol_candidates:
                    if m and s not in seen:
                        fp = mol_to_fp(m)
                        if max_similarity(fp, existing_fps) < diversity_threshold:
                            final_smi = s
                            existing_fps.append(fp)
                            break
                # valid + unique
                if final_smi is None:
                    for s, m in mol_candidates:
                        if m and s not in seen:
                            final_smi = s
                            existing_fps.append(mol_to_fp(m))
                            break
                # valid
                if final_smi is None:
                    for s, m in mol_candidates:
                        if m:
                            final_smi = s
                            existing_fps.append(mol_to_fp(m))
                            break
                # fallback
                if final_smi is None:
                    final_smi = candidates[0]

                seen.add(final_smi)
                gen_smiles_all.append(final_smi)

    # Compute metrics
    unique_gen_smiles = list(set(gen_smiles_all))

    validity   = 100.0 * sum(1 for s in unique_gen_smiles if is_valid_smiles(s)) / len(unique_gen_smiles)
    uniqueness = 100.0 * len(unique_gen_smiles) / len(gen_smiles_all)
    novelty    = 100.0 * sum(1 for s in unique_gen_smiles if s not in real_smiles_all) / len(unique_gen_smiles)
    diversity  = compute_diversity(unique_gen_smiles)

    metrics = {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "diversity": diversity
    }

    return real_smiles_all, gen_smiles_all, metrics