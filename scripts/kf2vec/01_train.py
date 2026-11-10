"""
Function: runs the entire kf2vec pipeline
1. precompute k-mer with jellyfish
2. divide tree into subtrees with TreeCluster
3. precompute distance from tree
4. perform model training
    - classifier classifies which tree it belongs.
    - train set: trains each tree.

Usage
-----
    python scripts/kf2vec/01_train.py --domain ar53 --split-file <split.hdf5> --output-folder <output_folder> --input-fasta <representative_genomes_dir> --conda kf2vec
"""
from __future__ import annotations

# Standard Import
import argparse
import numpy as np
from pathlib import Path

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5, folder
from src.tetrasphere import kf2vec
from src.system import create_logger

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain",             type=str,   required=True, choices=["ar53", "bac120"], help="which domain to process.")
    parser.add_argument("--split-file",         type=Path,  required=True, help="the matching split file.")
    parser.add_argument("--output-folder",      type=Path,  required=True, help="output path to all kf2vec results.")
    parser.add_argument("--input-fasta",        type=Path,  required=True, help="where all genomes are stored at.")
    parser.add_argument("--release",            default=232, type=int,  help="which GTDB release.")
    
    # kf2vec parameters
    parser.add_argument("--k",                  default=7,      type=int,   help="k-mer length for kf2vec's own canonical composition input.")
    parser.add_argument("--subtree-size",       default=850,    type=int,   help="TreeCluster target subtree size.")
    parser.add_argument("--classifier-epochs",  default=2000,   type=int,   help="kf2vec train_classifier max epochs.")
    parser.add_argument("--classifier-batch",   default=16,     type=int,   help="kf2vec train_classifier batch size.")
    parser.add_argument("--classifier-lr",      default=1e-5,   type=float, help="kf2vec train_classifier learning rate.")
    parser.add_argument("--classifier-hidden",  default=2048,   type=int,   help="kf2vec train_classifier hidden layer size.")
    parser.add_argument("--distance-epochs",    default=8000,   type=int,   help="kf2vec train_model_set epoch count, per subtree.")
    parser.add_argument("--distance-batch",     default=16,     type=int,   help="kf2vec train_model_set batch size.")
    parser.add_argument("--distance-lr",        default=1e-5,   type=float, help="kf2vec train_model_set learning rate.")
    parser.add_argument("--distance-hidden",    default=2048,   type=int,   help="kf2vec train_model_set hidden layer size.")
    parser.add_argument("--distance-embed",     default=1024,   type=int,   help="kf2vec train_model_set embedding dimension.")

    parser.add_argument("--seed",           default=50,         type=int, help="random seed, for reproducibility.")
    parser.add_argument("--workers",        default=10,         type=int, help="thread count for genome staging.")
    parser.add_argument("--conda",          default=config.env_kf2vec, type=str, help="conda env where kf2vec is installed.")
    

    return parser.parse_args()  

# Main
def main() -> None:
    args = parse_args()
    logger = create_logger(f"benchmark_kf2vec_{args.domain}", output_log=config.root_log)

    # Step 0. Load train-val-eval .hdf5 sets (for consistency)
    data = hdf5.load(args.split_file)
    fold_names = sorted((k for k in data if k.startswith("fold_")), key=lambda k: int(k.split("_")[1]))
    eval_leaf_ids = data["eval"]["leaf_id"]
    fold_leaf_ids = {name: {"train": data[name]["train"]["leaf_id"], "val": data[name]["val"]["leaf_id"]} for name in fold_names}
    unique_genome_ids = np.unique(np.concatenate([eval_leaf_ids, *(ids for pair in fold_leaf_ids.values() for ids in pair.values())]))
 
    # Step 1. Create .kf for each genome
    logger.info("Step 1: creating .kf for each genome...")
    logger.info(f"{len(unique_genome_ids)} unique genomes across every fold + eval")
    folder_kf = args.output_folder / "kf"
    kf2vec.copy_genome(unique_genome_ids, args.input_fasta, folder_kf, workers=args.workers, logger=logger)
    kf2vec.precompute_frequencies(folder_kf, folder_kf, args.k, args.workers, logger, args.conda)

    # Step 2. Symlink the shared eval query .kf subset once
    logger.info("Step 2: creating symlink for each .kf in each fold...")
    for fold in fold_names:
        folder_train    = args.output_folder / f"{fold}" / "kf_train"
        folder_val      = args.output_folder / f"{fold}" / "kf_val"
        folder_eval     = args.output_folder / f"{fold}" / "kf_eval"
        folder.create_symlink(folder_kf, folder_train, logger, [f"{leaf_id}.kf" for leaf_id in fold_leaf_ids[fold]['train']])
        folder.create_symlink(folder_kf, folder_val, logger, [f"{leaf_id}.kf" for leaf_id in fold_leaf_ids[fold]['val']])
        folder.create_symlink(folder_kf, folder_eval, logger, [f"{leaf_id}.kf" for leaf_id in eval_leaf_ids])
        
        # Step 2.2 Tree pruning
        file_tree = config.gtdb_download / f"{args.domain}_r{args.release}.tree.gz"
        file_prunned_tree = args.output_folder / f"{fold}" / f"{args.domain}_r{args.release}_{fold}.subtree"
        kf2vec.prune_tree(file_tree, fold_leaf_ids[fold]['train'], file_prunned_tree, logger)
    
    # Step 3. Run per fold training
    for fold in fold_names:
        # Perform tree clustering
        logger.info(f"Step 3.1: Prunning tree for {fold}...")
        file_prunned_tree = args.output_folder / f"{fold}" / f"{args.domain}_r{args.release}_{fold}.subtree"
        file_subtree = kf2vec.divide_trees(file_prunned_tree, args.subtree_size, logger, args.conda)

        # Step 3.1: Report remaining genomes
        folder_train    = args.output_folder / f"{fold}" / "kf_train"
        n_train = len(list(folder_train.glob("*.kf")))
        kf2vec.check_coverage(file_subtree, n_train, logger)
        
        # Step 3.2: Compute distance
        logger.info(f"Step 3.2: Precomputing distance for {fold}...")
        kf2vec.precompute_distance(file_prunned_tree, file_subtree, logger, args.conda)

        # Step 3.3: Train the router classifier and the per-subtree distance models
        logger.info(f"Step 3.2: Training classifier for {fold}...")
        kf2vec.train_classifier(
            folder_train, file_subtree, args.output_folder / f"{fold}",
            args.classifier_epochs, args.classifier_hidden, args.classifier_batch, args.classifier_lr, args.seed,
            logger, args.conda
        )
        logger.info(f"Step 3.2: Training distance regressor {fold}...")
        kf2vec.train_distance(
            folder_train, file_subtree, file_prunned_tree.parent, args.output_folder / f"{fold}",
            args.distance_epochs, args.distance_hidden, args.distance_embed, args.distance_batch, args.distance_lr, args.seed,
            logger, args.conda,
        )

if __name__ == "__main__":
    main()
