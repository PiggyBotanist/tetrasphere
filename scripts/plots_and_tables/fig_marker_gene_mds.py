"""
Function: 
    1. extracts specific genes (e.g. 16S/23S rRNA, gyrA, topoisomerase) from RefSeq genomes.
    2. embeds each gene copy through the fragment-curriculum tetrasphere checkpoint.
    3. projects to 2D via MDS on cosine distance, and plots one figure.

Usage
-----
    python scripts/plots_and_tables/fig_marker_gene_mds.py --domain bac120 \\
        --species "Escherichia coli" "Salmonella enterica" "Bacillus subtilis" "Listeria monocytogenes" \\
        --gene "16S ribosomal RNA" "23S ribosomal RNA" "DNA gyrase, subunit A"
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
from src.io import gff, fasta
from src.tetrasphere.visualization import reduce_to_2d_mds

REFSEQ_DOMAIN = {"ar53": "Archaea", "bac120": "Bacteria"}
GENE_FEATURE_TYPES = ["rRNA", "CDS", "tRNA"]

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Domain, species, genes
    parser.add_argument("--domain",   required=True, type=str, choices=["ar53", "bac120"], help="which domain (selects the fragment checkpoint and filters the RefSeq species pool).")
    parser.add_argument("--species",  required=True, type=str, nargs="+", help="organism names to query in the RefSeq manifest, e.g. \"Escherichia coli\".")
    parser.add_argument("--gene",     required=True, type=str, nargs="+", help="gene product-name substrings (or exact gene symbols) to extract, e.g. \"16S ribosomal RNA\" gyrA.")

    # Checkpoint
    parser.add_argument("--fold",     default=1, type=int, help="which fold's checkpoint to embed with.")
    parser.add_argument("--epoch",    default=None, type=int, help="checkpoint epoch (default: the latest one saved).")
    parser.add_argument("--device",   default=None, type=str, help="e.g. 'cuda:0'; auto-selects cuda if unset.")

    parser.add_argument("--max-copies", default=1, type=int, help="cap on gene copies plotted per (species, gene); extras are dropped (takes the first --max-copies hits in GFF order). Set to 0 for no cap. At the default of 1, there's exactly one point per (species, gene) -- marker shape = gene, color = species, no text labels (see main()). Above 1, falls back to the old per-copy text-labeled style, since shape/color alone can't distinguish multiple copies within the same (species, gene).")
    parser.add_argument("--genome-pivot", dest="genome_pivot", action="store_true", default=True, help="also embed each species' FULL GENOME (through the genome-scale path, Manifold directly -- not the Denoiser, which was never trained on multi-Mb input) as a small solid point with no outline, for comparison against its gene fragments. On by default; --no-genome-pivot to skip.")
    parser.add_argument("--no-genome-pivot", dest="genome_pivot", action="store_false")

    # MDS + output
    parser.add_argument("--seed",     default=42, type=int, help="MDS random seed.")
    parser.add_argument("--output",   default=None, type=Path, help="output .png path; default result/figure/<domain>_gene_mds.png.")
    return parser.parse_args()

# =================================================================
# Pipeline steps
# =================================================================
def resolve_species(domain: str, species_names: list[str]) -> dict[str, dict[str, str]]:
    """Map each queried species name -> {accession, fasta_filename, gff_filename} of one RefSeq
        - Unresolved species are skipped with a warning.
    """
    manifest = pd.read_csv(config.refseq_manifest, sep="\t", compression="gzip")
    manifest = manifest.rename(columns={"#assembly_accession": "assembly_accession"})
    manifest["domain"] = manifest["domain"].str.removeprefix("domain:")
    manifest["species"] = manifest["species"].str.removeprefix("species:")
    manifest = manifest[manifest["domain"] == REFSEQ_DOMAIN[domain]]

    resolved = {}
    for name in species_names:
        match = manifest[manifest["species"].str.lower() == name.lower()]
        if match.empty:
            print(f"  WARNING: species {name!r} not found among RefSeq {REFSEQ_DOMAIN[domain]} genomes -- skipping")
            continue
        if (match["refseq_category"] == "reference genome").any():
            match = match[match["refseq_category"] == "reference genome"]
        row = match.sort_values("assembly_accession").iloc[0]  # deterministic pick when >1 strain is flagged "reference genome"
        resolved[name] = {"accession": row["assembly_accession"], "fasta_filename": row["fasta_filename"], "gff_filename": row["gff_filename"]}
    if not resolved:
        raise ValueError(f"none of {species_names} resolved against the RefSeq {REFSEQ_DOMAIN[domain]} manifest")
    return resolved

def find_gene_copies(gene_query: str, features: pd.DataFrame) -> pd.DataFrame:
    """
    Query all genes with the name.
    """
    hits = features[features["gene"].str.lower() == gene_query.lower()] if "gene" in features.columns else features.iloc[0:0]
    if hits.empty:
        hits = features[features["product"].str.contains(gene_query, case=False, na=False, regex=False)]
    return hits

# Main
def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Resolving {len(args.species)} species against the RefSeq {REFSEQ_DOMAIN[args.domain]} manifest...")
    genomes = resolve_species(args.domain, args.species)

    species_list, genes, sequences, labels = [], [], [], []
    genome_species, genome_sequences = [], []
    for species_name, info in genomes.items():
        gff_path = config.refseq_metadata / info["gff_filename"]
        fasta_path = config.refseq_sequence / info["fasta_filename"]
        features = gff.load(gff_path, feature_types=GENE_FEATURE_TYPES)
        records = fasta.load(fasta_path, to_dict=True)

        if args.genome_pivot:
            genome_species.append(species_name)
            genome_sequences.append("".join(str(r.seq) for r in records.values()))

        for gene_query in args.gene:
            hits = find_gene_copies(gene_query, features)
            if hits.empty:
                print(f"  {species_name}: no match for {gene_query!r}")
                continue
            n_total = len(hits)
            if args.max_copies:
                hits = hits.head(args.max_copies)
            for copy_idx, (_, row) in enumerate(hits.iterrows(), start=1):
                seq = gff.extract_sequence(row, records)
                sequences.append(seq)
                species_list.append(species_name)
                genes.append(gene_query)
                if args.max_copies != 1:  # only needed to disambiguate copies when there's >1 per group
                    labels.append(f"{species_name} #{copy_idx} ({len(seq)/1000:.1f}kb)")
            details = list(zip(hits["type"], (hits["end"] - hits["start"] + 1)))
            cap_note = f" (capped from {n_total})" if n_total > len(hits) else ""
            print(f"  {species_name} / {gene_query!r}: {len(hits)} copy(s){cap_note} -- (type, length): {details}")

    if not sequences:
        raise ValueError("no gene copies matched anything -- check --species/--gene against the GFF product/gene fields")

    print("Loading fragment-curriculum checkpoint...")
    model, kmer = _embedding.load_model(args.domain, args.fold, args.epoch, device)

    print(f"Embedding {len(sequences)} gene sequences...")
    embeddings = _embedding.embed_sequences(model, kmer, sequences, device)

    n_genome = len(genome_sequences)
    if n_genome:
        # Genome-scale path: Manifold directly (model.manifold, frozen+shared with the standalone
        # genome-stage checkpoint -- see _embedding.py's docstring), NOT embed_sequences()'s
        # Denoiser->Manifold path, which was only ever trained on short fragments, not multi-Mb
        # genomes. include_length=False to match: genome-scale input has no length feature.
        print(f"Embedding {n_genome} full genomes (genome-scale path)...")
        genome_vectors = np.stack([kmer.vectorize(seq, include_length=False, transform_kmer=_embedding.TRANSFORM) for seq in genome_sequences])
        with torch.no_grad():
            genome_embeddings = model.manifold.encode(torch.from_numpy(genome_vectors).float().to(device)).cpu().numpy()
        embeddings = np.concatenate([embeddings, genome_embeddings], axis=0)

    print("Projecting to 2D via MDS (cosine distance)...")
    coords = reduce_to_2d_mds(embeddings, seed=args.seed)

    coords = np.asarray(coords)
    coords, genome_coords = (coords[:-n_genome], coords[-n_genome:]) if n_genome else (coords, np.empty((0, 2)))
    fig, ax = plt.subplots(figsize=(11, 9))

    if args.max_copies == 1:
        # One point per (species, gene): marker shape = gene, color = species, no text labels needed --
        # same-shape points clustering together (regardless of color) means gene identity dominates;
        # same-color points clustering together (regardless of shape) means species identity dominates.
        MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*"]
        unique_genes = list(dict.fromkeys(genes))
        unique_species = list(genomes.keys())  # every RESOLVED species (not just ones with a gene match), so genome-pivot points always have a color too
        gene_marker = {gene: MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, gene in enumerate(unique_genes)}
        species_cmap = plt.get_cmap("tab20" if len(unique_species) <= 20 else "gist_ncar")
        species_color = {sp: species_cmap(i / max(len(unique_species) - 1, 1)) for i, sp in enumerate(unique_species)}

        for gene in unique_genes:
            mask = np.array([g == gene for g in genes])
            colors = [species_color[species_list[i]] for i in np.nonzero(mask)[0]]
            ax.scatter(coords[mask, 0], coords[mask, 1], marker=gene_marker[gene], c=colors, s=90,
                       edgecolors="black", linewidths=0.5, zorder=3)

        if n_genome:
            # Full-genome pivot: small solid circle, no outline, colored by species -- visually distinct
            # from the (larger, black-edged) 16S circle so it never reads as just another gene point.
            genome_colors = [species_color[sp] for sp in genome_species]
            ax.scatter(genome_coords[:, 0], genome_coords[:, 1], marker="o", c=genome_colors, s=35,
                       edgecolors="none", zorder=2)

        gene_handles = [plt.Line2D([0], [0], marker=m, linestyle="", color="0.3", markersize=10, label=g) for g, m in gene_marker.items()]
        if n_genome:
            gene_handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color="0.3", markersize=6, label="full genome"))
        species_handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=c, markersize=10, label=_plotting.abbreviate_species(sp)) for sp, c in species_color.items()]
        gene_legend = ax.legend(handles=gene_handles, title="gene (shape)", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, labelcolor="black")
        ax.add_artist(gene_legend)
        species_legend = ax.legend(handles=species_handles, title="species (color)", loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=8, labelcolor="black")
        extra_artists = (gene_legend, species_legend)
    else:
        # Multiple copies per (species, gene): fall back to the old style -- color = gene, text label =
        # species + copy index + length, since shape/color alone can't disambiguate individual copies.
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        gene_color = {gene: palette[i % len(palette)] for i, gene in enumerate(dict.fromkeys(genes))}
        point_colors = [gene_color[g] for g in genes]
        texts = _plotting.scatter_with_labels(ax, coords[:, 0], coords[:, 1], labels, point_colors, size=30, fontsize=15, style="italic")
        _plotting.declutter_labels(texts, coords[:, 0], coords[:, 1], ax)
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=gene) for gene, color in gene_color.items()]
        ax.legend(handles=handles, title="gene", loc="best", fontsize=9, labelcolor="black")
        extra_artists = ()

    _plotting.style_axes(ax, "MDS dimension 1", "MDS dimension 2")
    fig.tight_layout()

    output_path = args.output or (config.result_figure / f"{args.domain}_gene_mds.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", bbox_extra_artists=extra_artists)
    print(f"Saved -> {output_path}")

if __name__ == "__main__":
    main()
