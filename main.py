import argparse
import os

from inference.generator_use import generate_smiles_from_features
from training.encoder_train import train
from inference.encoder_use import embed_samples
from training.generator_train import train_conditional_molt5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeneContrastiveModel training script")

    parser.add_argument("--train_conditional_encoder", action="store_true", help="Flag to train the encoder")
    parser.add_argument("--use_conditional_encoder", action="store_true", help="Flag to use the encoder")
    parser.add_argument("--train_molecular_generator", action="store_true", help="Flag to train the generator")
    parser.add_argument("--use_molecular_generator", action="store_true", help="Flag to use the generator")

    parser.add_argument("--encoder_data_path", type=str, default="./datasets/data/mcf7.csv",
                        help="Path to CSV expression data")
    parser.add_argument("--encoder_model_path", type=str, default="./checkpoints/conditional_encoder.pt",
                        help="Path to trained conditional encoder model")
    parser.add_argument("--statistics_path", type=str, default="./datasets/data/mcf7_statistics",
                        help="Path to save/load gene statistics")
    parser.add_argument("--embed_input_path", type=str, default="./datasets/data/mcf7.csv",
                        help="Path to embed CSV")
    parser.add_argument("--embed_output_path", type=str, default="./datasets/data/embed_mcf7.csv",
                        help="Path save embed CSV")

    parser.add_argument("--base_std", type=float, default=0.02)
    parser.add_argument("--dropout_rate", type=float, default=0.15)
    parser.add_argument("--scaling_range", type=float, nargs=2, default=(0.85, 1.15))
    parser.add_argument("--p_noise", type=float, default=0.0)
    parser.add_argument("--p_structured_noise", type=float, default=0.0)
    parser.add_argument("--p_dropout", type=float, default=1.0)
    parser.add_argument("--p_scaling", type=float, default=0.0)
    parser.add_argument("--max_augments", type=int, default=2)

    parser.add_argument("--encoder_batch_size", type=int, default=256)
    parser.add_argument("--encoder_input_dim", type=int, default=978)
    parser.add_argument("--encoder_num_features", type=int, default=320)
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--encoder_hidden_dim", type=int, default=64)
    parser.add_argument("--encoder_latent_size", type=int, default=128)
    parser.add_argument("--encoder_projection_dim", type=int, default=128)

    parser.add_argument("--encoder_lr", type=float, default=3e-4)
    parser.add_argument("--encoder_epochs", type=int, default=100)
    parser.add_argument("--encoder_temperature", type=float, default=0.2)

    parser.add_argument("--t5_path", type=str, default="./MolT5-small-caption2smiles",
                        help="Path to pretrained MolT5 model")
    parser.add_argument("--generator_data_path", type=str, default="./datasets/data/processed_mcf7.csv",
                        help="Path to CSV expression data")
    parser.add_argument("--cond_mode", type=str, default="vector")

    parser.add_argument("--generator_train_batch_size", type=int, default=32)
    parser.add_argument("--generator_test_batch_size", type=int, default=16)

    parser.add_argument("--cond_tokens", type=int, default=32)
    parser.add_argument("--generator_lr", type=float, default=1e-4)
    parser.add_argument("--generator_epochs", type=int, default=200)

    parser.add_argument("--con_t5_model_path", type=str, default="./checkpoints/cond_model.pt",
                        help="Path to generator inference condition vector CSV")
    parser.add_argument("--gen_feature_path", type=str, default="./datasets/data/gen_test.csv",
                        help="Path to CSV condition data")
    parser.add_argument("--gen_smiles_save_path", type=str, default="./datasets/data/generated_smiles.csv",
                        help="Path to CSV smiles generated output")

    parser.add_argument("--gen_num_aug", type=int, default=99)
    parser.add_argument("--gen_noise_std", type=float, default=0.1)
    parser.add_argument("--num_smiles_per_sample", type=int, default=10)

    parser.add_argument("--save_dir", type=str, default="./checkpoints")


    args = parser.parse_args()

    ## train conditional encoder ##

    if args.train_conditional_encoder:
        os.makedirs(args.save_dir, exist_ok=True)
        train(
            data_path=args.encoder_data_path,
            statistics_path=args.statistics_path,
            base_std=args.base_std,
            dropout_rate=args.dropout_rate,
            scaling_range=args.scaling_range,
            p_noise=args.p_noise,
            p_structured_noise=args.p_structured_noise,
            p_dropout=args.p_dropout,
            p_scaling=args.p_scaling,
            max_augments=args.max_augments,
            batch_size=args.encoder_batch_size,
            input_dim=args.encoder_input_dim,
            num_features=args.encoder_num_features,
            num_transformer_layers=args.num_transformer_layers,
            hidden_dim=args.encoder_hidden_dim,
            latent_size=args.encoder_latent_size,
            projection_dim=args.encoder_projection_dim,
            lr=args.encoder_lr,
            epochs=args.encoder_epochs,
            temperature=args.encoder_temperature,
            save_dir=args.save_dir
        )

    ##################################################

    ## use conditional encoder ##

    if args.use_conditional_encoder:
        embed_samples(
            data_path=args.embed_input_path,
            encoder_path=args.encoder_model_path,
            output_csv=args.embed_output_path,
            input_dim=args.encoder_input_dim,
            num_features=args.encoder_num_features,
            hidden_dim=args.encoder_hidden_dim,
            latent_size=args.encoder_latent_size,
            batch_size=args.encoder_batch_size
        )

    ##################################################

    ## train molecular generator ##

    if args.train_molecular_generator:
        cond_model, train_loader, test_loader, tokenizer = train_conditional_molt5(
            csv_path=args.generator_data_path,
            t5_path=args.t5_path,
            train_batch_size=args.generator_train_batch_size,
            test_batch_size=args.generator_test_batch_size,
            cond_tokens=args.cond_tokens,
            cond_mode=args.cond_mode,
            lr=args.generator_lr,
            epochs=args.generator_epochs,
            save_dir=args.save_dir
        )

    ##################################################

    ## use molecular generator ##

    if args.use_molecular_generator:
        generate_smiles_from_features(
            feature_csv=args.gen_feature_path,
            cond_model_pt=args.con_t5_model_path,
            t5_path=args.t5_path,
            output_csv=args.gen_smiles_save_path,
            cond_tokens=args.cond_tokens,
            cond_mode=args.cond_mode,
            num_augment=args.gen_num_aug,
            noise_std=args.gen_noise_std,
            num_smiles_per_sample=args.num_smiles_per_sample
        )

    ##################################################