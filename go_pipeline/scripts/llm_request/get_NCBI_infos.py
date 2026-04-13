import json
import gzip
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def load_gene_symbol_to_id_mapping(gene_info_file):
    """Load gene symbol to GeneID mapping from Homo_sapiens.gene_info.gz."""
    symbol_to_id = {}
    try:
        with gzip.open(gene_info_file, "rt", encoding="utf-8") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    tax_id, gene_id, symbol = parts[0], parts[1], parts[2]
                    if tax_id == "9606" and symbol != "-" and symbol.strip():
                        symbol_to_id[symbol] = gene_id
    except Exception as e:
        print(f"Error loading gene symbol mapping: {e}")
    return symbol_to_id


def load_gene_id_to_summary_mapping(gene_summary_file):
    """Load GeneID to summary mapping from gene_summary.gz."""
    id_to_summary = {}
    try:
        with gzip.open(gene_summary_file, "rt", encoding="utf-8") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    tax_id, gene_id, source, summary = parts[0], parts[1], parts[2], parts[3]
                    if tax_id == "9606" and gene_id.strip() and summary.strip() and summary != "-":
                        id_to_summary[gene_id] = summary
    except Exception as e:
        print(f"Error loading gene summaries: {e}")
    return id_to_summary


def load_ncbi_gene_summaries(gene_summary_file, gene_info_file):
    """Combine symbol->ID and ID->summary into symbol->summary mapping."""
    if not os.path.exists(gene_info_file):
        print(f"{gene_info_file} not found")
        return {}

    symbol_to_id = load_gene_symbol_to_id_mapping(gene_info_file)
    id_to_summary = load_gene_id_to_summary_mapping(gene_summary_file)

    symbol_to_summary = {}
    for symbol, gene_id in symbol_to_id.items():
        if gene_id in id_to_summary:
            symbol_to_summary[symbol] = id_to_summary[gene_id]

    return symbol_to_summary


def enhance_ranking_json_with_summaries(ranking_file, gene_summaries):
    """Add NCBI gene summaries to ranking JSON (including representatives)."""
    try:
        with open(ranking_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_genes = 0
        enhanced = 0
        missing = 0

        for term in data.get("ranking", []):
            gene_symbols = term.get("gene_symbols", [])
            enhanced_dict = {}

            for symbol in gene_symbols:
                total_genes += 1
                if symbol in gene_summaries:
                    enhanced_dict[symbol] = gene_summaries[symbol]
                    enhanced += 1
                else:
                    enhanced_dict[symbol] = "No summary available"
                    missing += 1

            term["gene_symbols"] = enhanced_dict

        with open(ranking_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Error processing {ranking_file}: {e}")
        return False


def enhance_manifold_analysis_json_with_summaries(manifold_file, gene_summaries):
    """Add NCBI gene summaries to manifold analysis JSON."""
    try:
        with open(manifold_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_genes = 0
        enhanced = 0
        missing = 0

        for group in data.get("groups", []):

            if "representative" in group and "gene_symbols" in group["representative"]:
                gene_symbols = group["representative"]["gene_symbols"]
                if isinstance(gene_symbols, list):
                    enhanced_dict = {}
                    for symbol in gene_symbols:
                        total_genes += 1
                        if symbol in gene_summaries:
                            enhanced_dict[symbol] = gene_summaries[symbol]
                            enhanced += 1
                        else:
                            enhanced_dict[symbol] = "No summary available"
                            missing += 1
                    group["representative"]["gene_symbols"] = enhanced_dict

            for term in group.get("terms", []):
                if "gene_symbols" in term:
                    gene_symbols = term["gene_symbols"]
                    if isinstance(gene_symbols, list):
                        enhanced_dict = {}
                        for symbol in gene_symbols:
                            total_genes += 1
                            if symbol in gene_summaries:
                                enhanced_dict[symbol] = gene_summaries[symbol]
                                enhanced += 1
                            else:
                                enhanced_dict[symbol] = "No summary available"
                                missing += 1
                        term["gene_symbols"] = enhanced_dict

            for connection in group.get("connections_to_other_groups", []):

                if "source_term" in connection and "gene_symbols" in connection["source_term"]:
                    gene_symbols = connection["source_term"]["gene_symbols"]
                    if isinstance(gene_symbols, list):
                        enhanced_dict = {}
                        for symbol in gene_symbols:
                            total_genes += 1
                            if symbol in gene_summaries:
                                enhanced_dict[symbol] = gene_summaries[symbol]
                                enhanced += 1
                            else:
                                enhanced_dict[symbol] = "No summary available"
                                missing += 1
                        connection["source_term"]["gene_symbols"] = enhanced_dict

                if "target_term" in connection and "gene_symbols" in connection["target_term"]:
                    gene_symbols = connection["target_term"]["gene_symbols"]
                    if isinstance(gene_symbols, list):
                        enhanced_dict = {}
                        for symbol in gene_symbols:
                            total_genes += 1
                            if symbol in gene_summaries:
                                enhanced_dict[symbol] = gene_summaries[symbol]
                                enhanced += 1
                            else:
                                enhanced_dict[symbol] = "No summary available"
                                missing += 1
                        connection["target_term"]["gene_symbols"] = enhanced_dict

        with open(manifold_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Error processing {manifold_file}: {e}")
        return False


def process_signature_directories(base_path, gene_summaries):
    """Process all signature directories and update ranking and manifold JSON files."""
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Input path {base_path} does not exist")
        return

    total_ranking_files = 0
    total_manifold_files = 0

    sig_dirs = [
        d for d in base_path.iterdir()
        if d.is_dir() and not d.name.startswith("diluted_")
    ]

    for sig_dir in tqdm(sig_dirs, desc="Processing signatures"):
        if not sig_dir.is_dir():
            continue

        if sig_dir.name.startswith("diluted_"):
            continue

        dilution_dir = sig_dir / "Dilutions" / "fixed"
        if dilution_dir.exists():
            for file_path in dilution_dir.glob("*ranking*.json"):
                success = enhance_ranking_json_with_summaries(file_path, gene_summaries)
                if success:
                    total_ranking_files += 1

        param_analysis_dir = sig_dir / "parameter_analysis"
        if param_analysis_dir.exists():
            for ontology_dir in param_analysis_dir.iterdir():
                if ontology_dir.is_dir():
                    ontology = ontology_dir.name
                    manifold_file = ontology_dir / f"manifold_analysis_{ontology}.json"

                    if manifold_file.exists():
                        success = enhance_manifold_analysis_json_with_summaries(
                            manifold_file, gene_summaries
                        )
                        if success:
                            total_manifold_files += 1


def main():
    parser = argparse.ArgumentParser(
        description="Add NCBI gene summaries to ranking and manifold analysis JSON files"
    )

    parser.add_argument(
        "--input_data", "-i",
        required=True,
        help="Path to directory containing signature folders"
    )

    parser.add_argument(
        "--gene_summary", "-g",
        default="data/NCBI/gene_summary.gz",
        help="Path to gene_summary.gz"
    )

    parser.add_argument(
        "--gene_info",
        default="data/Homo_sapiens.gene_info.gz",
        help="Path to Homo_sapiens.gene_info.gz"
    )

    args = parser.parse_args()

    gene_summaries = load_ncbi_gene_summaries(
        args.gene_summary,
        args.gene_info
    )

    if not gene_summaries:
        print("No gene summaries loaded. Exiting.")
        return

    process_signature_directories(args.input_data, gene_summaries)


if __name__ == "__main__":
    main()