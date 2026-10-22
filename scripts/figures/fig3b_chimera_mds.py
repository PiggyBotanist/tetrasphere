"""
Function: embeds real genomes and synthetic sequence chimeras through the fragment-curriculum
    tetrasphere checkpoint, projects everything to 2D via MDS on cosine distance, and plots one
    figure. Same-species points should sit close together, unrelated species should separate, and
    each chimera series -- increasing-ratio splices of real genomic DNA between two real parent
    genomes -- should trace a visibly non-linear path from parent A to parent B.

    Chimera construction: a length-proportional splice of real genomic DNA (the first ratio*target_len
    bp of parent A's concatenated sequence, spliced to the first (1-ratio)*target_len bp of parent B's),
    where target_len is fixed per pair (the mean of the two parents' own genome lengths) so every point
    in a series is the same total length -- only the A/B composition mix changes with ratio. This is a
    real synthetic chimeric contig, not a linear blend of k-mer vectors, so any curvature in the plotted
    path reflects the model's actual (nonlinear) response to a chimeric composition.

    Which two species get spliced together, and at what ratios, is a CHIMERA_RECIPES dict below (not a
    CLI flag) -- edit it directly to add/change pairs. --species only controls the extra, non-chimeric
    points plotted for context (both recipe parents are always included automatically).

Usage
-----
    python scripts/figures/fig3b_chimera_mds.py --domain bac120 
    python scripts/figures/fig3b_chimera_mds.py --domain ar53 
"""
from __future__ import annotations

# Standard Import
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _embedding
import _plotting

# Custom Import
from config import config
from src.io import hdf5, fasta
from src.tetrasphere.preprocess import strip_genomePrefix
from src.tetrasphere.visualization import reduce_to_2d_mds

# Hard coded for visualization (chimera mixture illustration)
CHIMERA_RECIPES = {
    "bac120": [
        #{"parent_a": "Salmonella enterica",         "parent_b": "Salmonella bongori",       "ratios": [0.25, 0.5, 0.75]},  # close: same genus, both enteric pathogens
        {"parent_a": "Mycobacterium tuberculosis",         "parent_b": "Bacillus subtilis",        "ratios": [0.25, 0.5, 0.75]},  # distant: Pseudomonadota vs Bacillota
    ],
    "ar53": [
        #{"parent_a": "Methanosarcina acetivorans",  "parent_b": "Methanosarcina mazei",     "ratios": [0.25, 0.5, 0.75]},  # close: same genus
        {"parent_a": "Methanomassiliicoccus luminyensis",  "parent_b": "Nitrosopumilus maritimus", "ratios": [0.25, 0.5, 0.75]},  # distant: Euryarchaeota vs Thaumarchaeota
    ],
}

# Hard coded for visualization (archaea/bacteria to include in visualization)
DEFAULT_SPECIES = {
    "ar53": [
        "Methanosarcina acetivorans",       # genus cluster (Methanosarcinia) -- chimera anchor
        "Methanosarcina mazei",             # genus cluster -- same genus as acetivorans
        "Methanosarcina barkeri",           # genus cluster -- same genus as acetivorans
        "Methanosphaera stadtmanae",        # Methanobacteriaceae; dominant human gut archaeon
        "Methanobacterium formicicum",      # Methanobacteriaceae
        "Methanomassiliicoccus luminyensis", # Methanomassiliicoccaceae; human gut archaeon ("7th order" methanogens)
        "Methanococcus maripaludis",        # Methanococci
        "Methanocaldococcus jannaschii",    # Methanococci
        "Halobacterium salinarum",          # Halobacteria; biotech model (bacteriorhodopsin)
        "Haloferax volcanii",               # Halobacteria; biotech model organism
        "Pyrococcus furiosus",              # Thermococci; source of Pfu DNA polymerase (PCR/diagnostics)
        "Thermococcus kodakarensis",        # Thermococci
        "Sulfolobus acidocaldarius",        # Thermoproteota; Thermoprotei
        "Saccharolobus solfataricus",       # Thermoproteota; Thermoprotei
        "Pyrobaculum aerophilum",           # Thermoproteota; Thermoprotei
        "Archaeoglobus fulgidus",           # Archaeoglobi; sulfate reducer
        "Nitrosopumilus maritimus",         # Thaumarchaeota; ammonia oxidizer, nitrogen cycle -- chimera anchor's distant partner
        "Nitrososphaera viennensis",        # Thaumarchaeota; ammonia oxidizer, soil/agricultural relevance
    ],
    "bac120": [
        "Salmonella enterica",         # genus cluster (Enterobacteriaceae); typhoid/enteric fever, salmonellosis -- chimera anchor
        "Salmonella bongori",          # genus cluster; reptile-associated, occasional human enteric infection
        "Escherichia coli",            # Enterobacteriaceae; UTI/sepsis/foodborne (covers the Shigella lineage too -- GTDB merges Shigella into E. coli)
        "Klebsiella pneumoniae",       # Enterobacteriaceae; ESKAPE, hospital-acquired pneumonia/sepsis, multidrug resistance
        "Yersinia pestis",             # Enterobacteriaceae; plague
        "Pseudomonas aeruginosa",      # Pseudomonadaceae; ESKAPE, hospital-acquired infection
        "Acinetobacter baumannii",     # Moraxellaceae; ESKAPE, multidrug-resistant hospital infection
        "Vibrio cholerae",             # Vibrionaceae; cholera
        "Helicobacter pylori",         # Campylobacterota; gastric ulcers/cancer
        "Staphylococcus aureus",       # Bacillota; ESKAPE (incl. MRSA), skin/bloodstream infection
        "Streptococcus pneumoniae",    # Bacillota; genus cluster; pneumonia/meningitis
        "Streptococcus pyogenes",      # Bacillota; genus cluster; strep throat/necrotizing fasciitis
        "Enterococcus faecalis",       # Bacillota; hospital-acquired infection, antibiotic resistance
        "Listeria monocytogenes",      # Bacillota; foodborne, meningitis
        "Clostridioides difficile",    # Bacillota_A; hospital-acquired diarrhea/colitis
        "Bacillus subtilis",           # Bacillota; non-pathogenic reference -- chimera anchor's distant partner
        "Mycobacterium tuberculosis",  # Actinomycetota; tuberculosis
        "Treponema pallidum",          # Spirochaetota; syphilis
    ],
}

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Domain and species
    parser.add_argument("--domain",   required=True, type=str, choices=["ar53", "bac120"], help="which GTDB domain (selects the fragment checkpoint and the species pool).")
    parser.add_argument("--species",  default=None, type=str, nargs="+", help="extra real species to plot for context (GTDB species names, e.g. \"Escherichia coli\"); default: a small built-in list. Chimera recipe parents are always included regardless.")
    parser.add_argument("--release",  default="232", type=str, help="GTDB release version, used to build the taxonomy filename.")

    # Checkpoint
    parser.add_argument("--fold",     default=1, type=int, help="which fold's checkpoint to embed with.")
    parser.add_argument("--epoch",    default=None, type=int, help="checkpoint epoch (default: the latest one saved).")
    parser.add_argument("--device",   default=None, type=str, help="e.g. 'cuda:0'; auto-selects cuda if unset.")

    # MDS + output
    parser.add_argument("--seed",     default=42, type=int, help="MDS random seed.")
    parser.add_argument("--output",   default=None, type=Path, help="output .png path; default result/figure/<domain>_chimera_mds.png.")
    return parser.parse_args()

# =================================================================
# Pipeline steps
# =================================================================
def resolve_species(domain: str, release: str, species_names: list[str]) -> dict[str, str]:
    """Map each queried species name -> its GTDB representative genome accession (stripped id), restricted to genomes actually present in the domain's precomputed feature file."""
    taxonomy_file = config.gtdb_download / f"{domain}_taxonomy_r{release}.tsv.gz"
    feature_file = config.gtdb_feature / f"{domain}_4mer_clr.hdf5"

    df = pd.read_csv(taxonomy_file, sep="\t", header=None, names=["accession", "taxonomy"], compression="gzip")
    df["accession"] = strip_genomePrefix(df["accession"].to_numpy())
    df["species"] = df["taxonomy"].str.extract(r"s__([^;]+)$")[0]

    feature_ids = set(hdf5.load(feature_file)["id"])
    df = df[df["accession"].isin(feature_ids)]

    resolved = {}
    for name in species_names:
        match = df[df["species"].str.lower() == name.lower()]
        if match.empty:
            raise ValueError(f"species {name!r} not found among {domain}'s representative genomes ({taxonomy_file})")
        resolved[name] = match.iloc[0]["accession"]
    return resolved

def load_concatenated_sequence(domain: str, accession: str) -> str:
    """Every contig of one representative genome, concatenated into a single string (record order as stored in the FASTA)."""
    path = config.gtdb_download / "representative_genomes" / f"{accession}_genomic.fna.gz"
    records = fasta.load(path)
    return "".join(str(r.seq) for r in records)

def splice_chimera(seq_a: str, seq_b: str, ratio: float, target_length: int) -> str:
    """First ratio*target_length bp of seq_a spliced to the first (1-ratio)*target_length bp of seq_b."""
    len_a = int(round(ratio * target_length))
    len_b = target_length - len_a
    return seq_a[:len_a] + seq_b[:len_b]

# Main
def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    recipes = CHIMERA_RECIPES[args.domain]

    species_names = list(dict.fromkeys((args.species or DEFAULT_SPECIES[args.domain]) + [n for r in recipes for n in (r["parent_a"], r["parent_b"])]))
    print(f"Resolving {len(species_names)} species against {args.domain} taxonomy...")
    accessions = resolve_species(args.domain, args.release, species_names)

    print("Loading genome sequences...")
    sequences = {name: load_concatenated_sequence(args.domain, acc) for name, acc in accessions.items()}
    for name, seq in sequences.items():
        print(f"  {name}: {len(seq):,} bp")

    print("Loading fragment-curriculum checkpoint...")
    model, kmer = _embedding.load_model(args.domain, args.fold, args.epoch, device)

    # Build the full point list: real species (group="species") + chimera series (group=recipe index)
    labels, groups, plot_sequences, series_id, series_ratio = [], [], [], [], []
    for name in species_names:
        labels.append(name); groups.append("species"); plot_sequences.append(sequences[name]); series_id.append(-1); series_ratio.append(None)

    for i, recipe in enumerate(recipes):
        seq_a, seq_b = sequences[recipe["parent_a"]], sequences[recipe["parent_b"]]
        target_length = (len(seq_a) + len(seq_b)) // 2
        for ratio in recipe["ratios"]:
            chimera_seq = splice_chimera(seq_a, seq_b, ratio, target_length)
            label = f"{int(ratio*100)}% {recipe['parent_a']} + {int((1-ratio)*100)}% {recipe['parent_b']}"
            labels.append(label); groups.append(f"chimera_{i}"); plot_sequences.append(chimera_seq); series_id.append(i); series_ratio.append(ratio)

    print(f"Embedding {len(plot_sequences)} sequences ({len(species_names)} real + {len(plot_sequences)-len(species_names)} chimera)...")
    embeddings = _embedding.embed_sequences(model, kmer, plot_sequences, device)

    print("Projecting to 2D via MDS (cosine distance)...")
    coords = reduce_to_2d_mds(embeddings, seed=args.seed)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 9))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    species_color = {name: palette[i % len(palette)] for i, name in enumerate(species_names)}

    species_mask = np.array([g == "species" for g in groups])
    chimera_mask = ~species_mask
    texts = []
    texts += _plotting.scatter_with_labels(ax, coords[species_mask, 0], coords[species_mask, 1], [l for l, m in zip(labels, species_mask) if m], [species_color[l] for l, m in zip(labels, species_mask) if m], size=40, fontsize=15, style="italic")
    texts += _plotting.scatter_with_labels(ax, coords[chimera_mask, 0], coords[chimera_mask, 1], [l for l, m in zip(labels, chimera_mask) if m], "0.5", size=20, fontsize=7, style="normal")

    for i, recipe in enumerate(recipes):
        series_mask = [g == f"chimera_{i}" for g in groups]
        a_idx, b_idx = labels.index(recipe["parent_a"]), labels.index(recipe["parent_b"])
        path_points = np.vstack([coords[b_idx], coords[np.array(series_mask)], coords[a_idx]])
        t = np.array([0.0] + [r for r, m in zip(series_ratio, series_mask) if m] + [1.0])
        order = np.argsort(t)
        path_sorted = path_points[order]
        ax.plot(path_sorted[:, 0], path_sorted[:, 1], color="black", linewidth=1, zorder=1, marker="o", markersize=3)
        ax.annotate("", xy=tuple(path_sorted[-1]), xytext=tuple(path_sorted[-2]), arrowprops=dict(arrowstyle="->", color="black", lw=1))

    label_x = np.concatenate([coords[species_mask, 0], coords[chimera_mask, 0]])
    label_y = np.concatenate([coords[species_mask, 1], coords[chimera_mask, 1]])
    _plotting.declutter_labels(texts, label_x, label_y, ax)
    _plotting.style_axes(ax, "MDS dimension 1", "MDS dimension 2")
    fig.tight_layout()

    output_path = args.output or (config.result_figure / f"{args.domain}_chimera_mds.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved -> {output_path}")

if __name__ == "__main__":
    main()
