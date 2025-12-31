# GCLR-AET5: A Framework for Robust Linkage of Gene Expression to *De Novo* Molecular Design via Contrastive Learning🚀

This repository contains the PyTorch implementation of **AET5**, which aims to generate potential bioactive molecules for specific diseases.

## Framework👀

![AET5](https://github.com/YuanZhikang-git/AET5/blob/main/picture/overa11.png)

## System Requirements👍

The source code was developed in Python 3.10 using PyTorch 2.5.0. The required Python dependencies are given below. AET5 is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There are no additional non-standard hardware requirements.

```
torch=2.5.0
numpy=2.2.2
pandas=2.2.3
rdkit~=2024.9.5
transformers=4.51.3
tqdm~=4.67.1
pandas~=2.2.3
```

## Datasets🐴

The `datasets` folder contains all experimental data used in AET5: [Drug-induced Signature](https://clue.io/data/CMap2020#LINCS2020), [SARS-Covid-2-Specific Signature](https://github.com/pth1993/DeepCE), [PC-Specific Signature](https://github.com/hzauzqy/TransGEM).

## Run AET5 on Our Data to Reproduce Results🏃

To train AET5, basic hyperparameter configurations are provided in the `main.py`.

We use a pretrained MolT5 model for model fine-tuning and drug generation. Please download the required MolT5 weights and related files and place them in the `./MolT5-small-caption2smiles` directory. The MolT5 model can be obtained from [MolT5 Hugging Face](https://huggingface.co/laituan245/molt5-small-caption2smiles).

For the conditional encoder, you can directly run the following command to train the expression embeddings. 

```
$ python main.py --train_conditional_encoder
```

After training, you can perform gene expression embedding inference using the command below.

```
$ python main.py --use_conditional_encoder
```

For the molecular generator, you can run the following command to perform fine-tuning.

```
$ python main.py --train_molecular_generator
```

After training, you can test the generation of new potential bioactive molecules using the corresponding inference command.

```
$ python main.py --use_molecular_generator
```

You may also attempt to generate molecules conditioned on expression profiles from other diseases by simply inverting the disease expression values and ensuring that the gene order matches the reference file `./datasets/tools/source_genes.csv`.
