from __future__ import annotations

# Standard Import
import gzip
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Custom Import
from src.system.shell import execute
from src.io import nwk
from src.tetrasphere.preprocess import strip_genomePrefix
from src.system import multithread

# =================================================================
# Preprocessing
# =================================================================
def copy_genome(leaf_ids: np.ndarray, input_folder: Path, output_folder: Path, workers: int, logger: logging.Logger) -> None:
    """
    Decompress and copy genome (prepare for jellyfish)
    
    Parameters
    ----------
    leaf_ids       : genome ids to stage (stripped of GB_/RS_).
    input_folder   : genome FASTA folder.
    output_folder  : where to write each {leaf_id}.fna.
    workers        : thread count for the symlinking pass.
    logger         : logs missing genomes and a final summary.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    def create_copy(leaf_id: str) -> None:
        source = input_folder / f"{leaf_id}_genomic.fna.gz"
        target = output_folder / f"{leaf_id}.fna"
        tmp = output_folder / f"{leaf_id}.tmp"

        if not source.exists():
            logger.warning(f"Missing genome: {source}")
            return

        if target.exists():
            logger.warning(f"Genome already exists, skipping: {target}")
            return

        # Handle partial download error
        with gzip.open(source, "rb") as f_in, open(tmp, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        # Only create final file after successful decompression
        tmp.replace(target)
                
    args = [(str(leaf_id),) for leaf_id in leaf_ids]
    names = [str(leaf_id) for leaf_id in leaf_ids]
    
    multithread(create_copy, args, names, workers, logger)
    logger.info(f"Staged {len(leaf_ids)} genomes -> {output_folder}") 

def precompute_frequencies(input_folder: Path, output_folder: Path, k: int, workers: int, logger: logging.Logger, conda_env: str | None = None) -> None:
    """
    runs > kf2vecget_frequencies (writes normalized .kf file per genome in folder_input)

    Parameters
    ----------
    input_folder  : genome sequences (.fastq/.fq/.fa/.fna/.fasta).
    output_folder : where to write each genome's .kf file.
    k             : k-mer length.
    workers       : jellyfish thread count.
    logger        : logs the underlying kf2vec command.
    conda_env     : conda env where kf2vec is installed.
    """
    execute(
        ["kf2vec", "get_frequencies",
         "-input_dir", str(input_folder), "-output_dir", str(output_folder),
         "-k", str(k), "-p", str(workers)],
        logger, conda_env=conda_env,
    )
    
def prune_tree(input_nwk: Path, leaf_ids: np.ndarray, output_path: Path, logger: logging.Logger) -> None:
    """
    Prune the full GTDB tree down to just the backbone leaves; writes a .nwk for kf2vec's -tree input.

    Parameters
    ----------
    input_nwk           : full .nwk tree.
    leaf_ids            : genome ids to keep.
    output_path         : where to write the pruned tree.
    logger              : logs how many leaves were kept.
    """
    tree = nwk.load(input_nwk)
    all_labels = np.array([t.label for t in tree.taxon_namespace], dtype=object)
    stripped   = strip_genomePrefix(all_labels)
    keep_labels = all_labels[np.isin(stripped, leaf_ids)].tolist()

    tree.retain_taxa_with_labels(keep_labels)
    for leaf in tree.leaf_node_iter():
        leaf.taxon.label = strip_genomePrefix(np.array([leaf.taxon.label], dtype=object))[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path=str(output_path), schema="newick")
    logger.info(f"Pruned tree -> {output_path} ({len(keep_labels)}/{len(leaf_ids)} backbone leaves found)")

def divide_trees(input_nwk: Path, subtree_size: int, logger: logging.Logger, conda_env: str | None = None) -> Path:
    """
    runs > kf2vec divide_tree (splits tree.nwk into subtrees; returns the resulting .subtrees file)

    Parameters
    ----------
    input_nwk     : backbone phylogeny (.newick/.nwk).
    subtree_size  : TreeCluster target subtree size.
    logger        : logs the underlying kf2vec command.
    conda_env     : conda env where kf2vec is installed.
    """
    execute(
        ["kf2vec", "divide_tree", "-tree", str(input_nwk), "-size", str(subtree_size)],
        logger, conda_env=conda_env,
    )
    return input_nwk.with_suffix(".subtrees")

def check_coverage(file_subtrees: Path, n_backbone: int, logger: logging.Logger) -> None:
    """
    Warn if divide_tree dropped genomes (kf2vec excludes ungrouped leaves during "TreeCluster".

    Parameters
    ----------
    file_subtrees  : .subtrees file from kf2vec divide_tree (space-separated, "genome clade" columns).
    n_backbone     : expected genome count (the backbone .kf/.fasta files divide_tree was run on).
    logger         : logs information.
    """
    df = pd.read_csv(file_subtrees, sep=" ", header=0)
    n_covered = len(df)

    if n_covered < n_backbone:
        logger.warning(f"divide_tree dropped {n_backbone - n_covered} of {n_backbone} genomes (likely TreeCluster -1/ungrouped leaves)")
    else:
        logger.info(f"divide_tree covered all {n_backbone} genomes")

    counts = df["clade"].value_counts().sort_index()
    logger.info(f"{len(counts)} subtrees, sizes: {counts.to_dict()}")

def precompute_distance(input_nwk: Path, input_subtrees: Path, logger: logging.Logger, conda_env: str | None = None) -> None:
    """
    runs > kf2vec get_distances (writes one {tree_stem}_subtree_{i}.di_mtrx file per subtree)

    Parameters
    ----------
    input_nwk      : backbone phylogeny (.newick/.nwk).
    input_subtrees : .subtrees file from run_divide_tree.
    logger         : logs the underlying kf2vec command.
    conda_env      : conda env where kf2vec is installed.
    """
    execute(
        ["kf2vec", "get_distances", "-tree", str(input_nwk), "-subtrees", str(input_subtrees)],
        logger, conda_env=conda_env,
    )

# =================================================================
# Library training
# =================================================================
def train_classifier(input_kf: Path, input_subtrees: Path, output_folder: Path, epochs: int, hidden: int, batch: int, lr: float, seed: int, logger: logging.Logger, conda_env: str | None = None) -> None:
    """
    runs > kf2vec train_classifier (writes classifier_model.ckpt to folder_output)

    Parameters
    ----------
    input_kf       : backbone .kf files, from precompute_frequencies().
    input_subtrees : .subtrees file from divide_trees().
    output_folder  : where to write classifier_model.ckpt.
    epochs         : max training epochs.
    hidden         : hidden layer size.
    batch          : batch size.
    lr             : starting learning rate.
    seed           : random seed, for reproducibility.
    logger         : logs the underlying kf2vec command.
    conda_env      : conda env where kf2vec is installed.
    """
    execute(
        ["kf2vec", "train_classifier",
         "-input_dir", str(input_kf), "-subtrees", str(input_subtrees),
         "-e", str(epochs), "-hidden_sz", str(hidden), "-batch_sz", str(batch), "-lr", str(lr), "-seed", str(seed),
         "-o", str(output_folder)],
        logger, conda_env=conda_env, log_runtime=True,
    )

def train_distance(input_kf: Path, input_subtrees: Path, input_distFolder: Path, output_folder: Path, epochs: int, hidden: int, embed: int, batch: int, lr: float, seed: int, logger: logging.Logger, conda_env: str | None = None) -> None:
    """
    runs > kf2vec train_model_set (writes model_subtree_{i}.ckpt to folder_output, one per subtree)

    Parameters
    ----------
    input_kf            : backbone .kf files, from precompute_frequencies().
    input_subtrees      : .subtrees file from divide_trees().
    input_distFolder    : folder holding the *.di_mtrx files, from run_get_distances.
    output_folder       : where to write each model_subtree_{i}.ckpt.
    epochs              : max training epochs, per subtree.
    hidden              : hidden layer size.
    embed               : embedding dimension.
    batch               : batch size.
    lr                  : starting learning rate.
    seed                : random seed, for reproducibility.
    logger              : logs the underlying kf2vec command.
    conda_env           : conda env where kf2vec is installed.
    """
    execute(
        ["kf2vec", "train_model_set",
         "-input_dir", str(input_kf), "-true_dist", str(input_distFolder), "-subtrees", str(input_subtrees),
         "-e", str(epochs), "-hidden_sz", str(hidden), "-embed_sz", str(embed), "-batch_sz", str(batch), "-lr", str(lr), "-seed", str(seed),
         "-o", str(output_folder)],
        logger, conda_env=conda_env, log_runtime=True,
    )
    
 # =================================================================
# Query evaluation
# =================================================================   
def classify(input_kf: Path, input_model: Path, output_folder: Path, seed: int, logger: logging.Logger, conda_env: str | None = None) -> Path:
    """
    runs > kf2vec classify (writes classes.out to output_folder: predicted subtree per query genome)

    Parameters
    ----------
    input_kf       : query .kf files (genomes not in the trained backbone, e.g. kf_val/kf_eval).
    input_model    : folder holding classifier_model.ckpt, from train_classifier().
    output_folder  : where to write classes.out.
    seed           : random seed, for reproducibility.
    logger         : logs the underlying kf2vec command.
    conda_env      : conda env where kf2vec is installed.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    execute(
        ["kf2vec", "classify",
         "-input_dir", str(input_kf), "-model", str(input_model),
         "-seed", str(seed), "-o", str(output_folder)],
        logger, conda_env=conda_env,
    )
    return output_folder / "classes.out"


def query(input_kf: Path, input_model: Path, input_classes: Path, output_folder: Path, seed: int, logger: logging.Logger, conda_env: str | None = None) -> None:
    """
    runs > kf2vec query
        - apples_input_di_mtrx_subtree_{i}.csv      distance of the genome
        - embedding_subtree_{i}.emb:                embedding vector of the genome

    Parameters
    ----------
    input_kf       : query .kf files, same folder passed to classify().
    input_model    : folder holding model_subtree_{i}.ckpt + embeddings_subtree_{i}.csv, from train_distance().
    input_classes  : folder holding classes.out, from classify().
    output_folder  : where to write per-subtree distance/embedding files.
    seed           : random seed, for reproducibility.
    logger         : logs the underlying kf2vec command.
    conda_env      : conda env where kf2vec is installed.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    execute(
        ["kf2vec", "query",
         "-input_dir", str(input_kf), "-model", str(input_model), "-classes", str(input_classes),
         "-seed", str(seed), "-o", str(output_folder)],
        logger, conda_env=conda_env,
    )

def load_query_predictions(output_folder: Path) -> pd.DataFrame:
    """
    Parse every query result in output_folder as .tsv (query_id, backbone_id, pred_dist).
    NOTE: kf2vec's own network is trained in sqrt-distance space, but its own query.py already undoes. 
    Therefore nothing is done here.

    Parameters
    ----------
    output_folder  : folder holding apples_input_di_mtrx_query_{i}.csv files, from query().
    """
    frames = []
    # Load and parse index
    for path in sorted(output_folder.glob("apples_input_di_mtrx_query_*.csv")):
        df = pd.read_csv(path, sep="\t", header=0, index_col=0)
        df.index.name = "query_id"
        long = df.reset_index().melt(id_vars="query_id", var_name="backbone_id", value_name="pred_dist")
        frames.append(long)

    if not frames:
        raise ValueError("Dataframe is empty!  Double Check.")

    combined = pd.concat(frames, ignore_index=True)
    return combined
