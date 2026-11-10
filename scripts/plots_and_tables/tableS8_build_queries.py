"""
Build one query FASTA per CAMI genome for the ANI-to-nearest-training-genome check (Table S8).

Usage
-----
    python scripts/plots_and_tables/tableS8_build_queries.py --taxdump /path/to/ncbi/taxdump \\
        --sample-filter-stats /path/to/table_sample_filter_stats.tsv --output-dir /path/to/scratch
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import fasta

DOMAINS = {2: "bacteria", 2157: "archaea", 2759: "eukaryota", 10239: "viruses"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taxdump", required=True, type=Path, help="directory containing NCBI taxdump's nodes.dmp and merged.dmp.")
    parser.add_argument("--sample-filter-stats", required=True, type=Path,
                         help="table_sample_filter_stats.tsv-style table with (category, sample) rows -- produced by the Figure 4B binning pipeline.")
    parser.add_argument("--comebin-archive", default=None, type=Path, help="default: config.cami_archive/comebin.")
    parser.add_argument("--output-dir", required=True, type=Path, help="scratch directory for query FASTAs + queries.tsv (can be large; not committed to the repo).")
    parser.add_argument("--workers", default=12, type=int)
    return parser.parse_args()


def load_taxonomy(taxdump: Path) -> tuple[dict[int, int], dict[int, int]]:
    parent = {}
    with open(taxdump / "nodes.dmp") as f:
        for line in f:
            tid, par = line.split("\t|\t", 2)[:2]
            parent[int(tid)] = int(par)
    merged = {}
    with open(taxdump / "merged.dmp") as f:
        for line in f:
            old, new = line.rstrip("\t|\n").split("\t|\t")
            merged[int(old)] = int(new)
    return parent, merged


def domain_of(tax_id: int, parent: dict[int, int], merged: dict[int, int]) -> str:
    tid = merged.get(tax_id, tax_id)
    while tid in parent and tid not in DOMAINS and parent[tid] != tid:
        tid = parent[tid]
    return DOMAINS.get(tid, "unknown")


def per_sample(category: str, sample: str, comebin_archive: Path) -> pd.DataFrame:
    """bp, contig count and evaluated flag (>= 5 kbp AND in COMEBin's contig set) per genome in one sample."""
    meta = pd.read_csv(config.cami_metadata / f"{sample}.tsv.gz", sep="\t", compression="gzip")
    meta = meta.rename(columns={"#anonymous_gsa_id": "ids"})[["ids", "genome_id", "tax_id"]]
    meta["genome_id"] = meta["genome_id"].astype(str)
    comebin = set(pd.read_csv(comebin_archive / sample / "comebin_res" / "embeddings.tsv.gz", sep="\t", index_col=0, usecols=[0]).index)
    lengths = {r.id: len(r.seq) for r in fasta.load(config.cami_sequence / f"{sample}.fasta.gz", length_filter=0)}
    meta = meta[meta["ids"].isin(lengths)].copy()
    meta["length"] = meta["ids"].map(lengths)
    meta["evaluated"] = (meta["length"] >= 5000) & meta["ids"].isin(comebin)
    g = meta.groupby("genome_id").agg(tax_id=("tax_id", "first"), bp=("length", "sum"), n_contigs=("ids", "size"), evaluated=("evaluated", "any")).reset_index()
    g["category"], g["sample"] = category, sample
    return g


def write_queries(sample: str, genomes: pd.DataFrame) -> None:
    meta = pd.read_csv(config.cami_metadata / f"{sample}.tsv.gz", sep="\t", compression="gzip")
    id_to_genome = dict(zip(meta["#anonymous_gsa_id"], meta["genome_id"].astype(str)))
    wanted = dict(zip(genomes["genome_id"], genomes["query"]))
    handles = {}
    for r in fasta.load(config.cami_sequence / f"{sample}.fasta.gz", length_filter=0):
        gid = id_to_genome.get(r.id)
        if gid in wanted:
            if gid not in handles:
                handles[gid] = open(wanted[gid], "w")
            handles[gid].write(f">{r.id}\n{r.seq}\n")
    for h in handles.values():
        h.close()


def main() -> None:
    args = parse_args()
    comebin_archive = args.comebin_archive or (config.cami_archive / "comebin")
    out = args.output_dir
    (out / "query").mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(args.sample_filter_stats, sep="\t")[["category", "sample"]]
    with ProcessPoolExecutor(args.workers) as pool:
        per = pd.concat(pool.starmap(per_sample, [(c, s, comebin_archive) for c, s in zip(samples["category"], samples["sample"])]), ignore_index=True)

    evaluated = per.groupby(["category", "genome_id"])["evaluated"].any()
    best = per.sort_values("bp", ascending=False).drop_duplicates(["category", "genome_id"]).set_index(["category", "genome_id"])
    best["evaluated"] = evaluated
    best = best.reset_index()

    parent, merged = load_taxonomy(args.taxdump)
    best["domain"] = [domain_of(int(t), parent, merged) for t in best["tax_id"]]
    best["query"] = [str(out / "query" / f"{c}__{g}.fa") for c, g in zip(best["category"], best["genome_id"])]

    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(write_queries, *zip(*[(s, grp) for s, grp in best.groupby("sample")])))

    best.to_csv(out / "queries.tsv", sep="\t", index=False)
    print(best.groupby(["domain", "evaluated"]).size())
    print(f"Saved -> {out / 'queries.tsv'} ({len(best)} genomes)")


if __name__ == "__main__":
    main()
