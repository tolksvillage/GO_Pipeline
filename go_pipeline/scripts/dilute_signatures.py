"""
Multi-Signature Dilution Analyzer

Creates a dilution analysis for multiple gene signatures by stepwise
addition of random genes from a GAF file.
"""

import random
import argparse
import os
from pathlib import Path
from tqdm import tqdm


def load_valid_genes(gene_file_path):
    """
    Loads the list of valid protein-coding genes from a file.
    """
    try:
        with open(gene_file_path, 'r', encoding='utf-8') as file:
            return set(file.read().split())
    except FileNotFoundError:
        return set()
    except Exception:
        return set()


def parse_gaf_file(gaf_file_path, valid_genes):
    """
    Parses a GAF file and extracts all unique gene symbols.
    Filters out ncRNAs, complexes, and other non-protein-coding entries.
    """
    gene_symbols = set()

    try:
        with open(gaf_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith('!'):
                    continue

                fields = line.strip().split('\t')

                # GAF format: column 3 (index 2) contains the gene symbol
                if len(fields) >= 3:
                    gene_symbol = fields[2].strip()
                    if gene_symbol and gene_symbol in valid_genes:
                        gene_symbols.add(gene_symbol)

    except FileNotFoundError:
        return set()
    except Exception:
        return set()

    return gene_symbols


def load_signature(signature_file):
    """
    Loads a gene signature from a text file.
    """
    signature = []
    try:
        with open(signature_file, 'r', encoding='utf-8') as file:
            for line in file:
                gene = line.strip()
                if gene:
                    signature.append(gene)
    except FileNotFoundError:
        return []
    except Exception:
        return []

    return signature


def create_random_pool(gene_symbols, exclude_genes, pool_size=1000):
    """
    Creates a pool of random genes that are not part of the original signature.
    """
    available_genes = list(gene_symbols - set(exclude_genes))

    if pool_size > len(available_genes):
        pool_size = len(available_genes)

    return random.sample(available_genes, pool_size)


def create_dilution_signatures(original_signature, random_pool, steps=10, genes_per_step=100, mode='cumulative'):
    """
    Creates dilution signatures by adding random genes stepwise.
    """
    dilution_signatures = []

    if mode == 'cumulative':
        if steps * genes_per_step > len(random_pool):
            steps = len(random_pool) // genes_per_step

        cumulative_random_genes = []

        for step in range(1, steps + 1):
            start_idx = (step - 1) * genes_per_step
            end_idx = step * genes_per_step

            new_random_genes = random_pool[start_idx:end_idx]
            cumulative_random_genes.extend(new_random_genes)

            diluted_signature = original_signature + cumulative_random_genes

            dilution_signatures.append({
                'step': step,
                'new_random_genes_count': len(new_random_genes),
                'total_random_genes_count': len(cumulative_random_genes),
                'total_genes_count': len(diluted_signature),
                'signature': diluted_signature,
                'mode': 'cumulative'
            })

    elif mode == 'fixed':
        if steps * genes_per_step > len(random_pool):
            steps = len(random_pool) // genes_per_step

        for step in range(1, steps + 1):
            start_idx = (step - 1) * genes_per_step
            end_idx = step * genes_per_step

            random_genes_for_this_step = random_pool[start_idx:end_idx]
            diluted_signature = original_signature + random_genes_for_this_step

            dilution_signatures.append({
                'step': step,
                'new_random_genes_count': len(random_genes_for_this_step),
                'total_random_genes_count': len(random_genes_for_this_step),
                'total_genes_count': len(diluted_signature),
                'signature': diluted_signature,
                'mode': 'fixed'
            })

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'cumulative' or 'fixed'.")

    return dilution_signatures


def get_signature_name(signature_path):
    """
    Extracts the filename from the signature file path, without extension.
    """
    return os.path.basename(signature_path).replace('.txt', '')


def save_signatures_for_signature(signature_name, dilution_signatures, output_dir, mode):
    """
    Saves all diluted signatures for a given original signature.
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for sig_info in dilution_signatures:
            step = sig_info['step']
            total_random_count = sig_info['total_random_genes_count']
            total_count = sig_info['total_genes_count']
            signature = sig_info['signature']

            if mode == 'cumulative':
                filename = f"diluted_{signature_name}_step{step:02d}_cumulative{total_random_count}_total{total_count}.txt"
            else:
                filename = f"diluted_{signature_name}_step{step:02d}_fixed{total_random_count}_total{total_count}.txt"

            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as file:
                for gene in signature:
                    file.write(f"{gene}\n")

    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Creates a dilution analysis for multiple gene signatures'
    )
    parser.add_argument(
        '--signatures', '-s',
        required=True,
        help='Path to a single signature file or a folder containing .txt signatures'
    )
    parser.add_argument(
        '--gaf', '-g',
        default='data/goa_human.gaf',
        help='Path to the GAF file used for random gene sampling (default: data/goa_human.gaf)'
    )
    parser.add_argument(
        '--output', '-o',
        default='analysis/results',
        help='Output directory (default: analysis/results)'
    )
    parser.add_argument(
        '--steps', '-n',
        type=int,
        default=10,
        help='Number of dilution steps (default: 20)'
    )
    parser.add_argument(
        '--pool-size', '-z',
        type=int,
        default=19570,
        help='Minimum size of the random gene pool per signature'
    )
    parser.add_argument(
        '--seed', '-r',
        type=int,
        help='Random seed for reproducible results'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='cumulative',
        choices=['cumulative', 'fixed', 'both'],
        help='Dilution mode: cumulative, fixed, or both (default: cumulative)'
    )
    parser.add_argument(
        '--genes', '-ge',
        default='data/all_genes.txt',
        help='Path to the file containing valid genes (default: data/all_genes.txt)'
    )
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    signature_paths = []
    signatures_info = []

    if os.path.isdir(args.signatures):
        signature_paths = sorted([
            os.path.join(args.signatures, f)
            for f in os.listdir(args.signatures)
            if f.endswith(".txt")
        ])
        if not signature_paths:
            return
    elif os.path.isfile(args.signatures):
        signature_paths = [args.signatures]
    else:
        return

    for signature_path in signature_paths:
        signature = load_signature(signature_path) or []
        signatures_info.append({
            'name': get_signature_name(signature_path),
            'path': signature_path,
            'signature': signature,
            'size': len(signature)
        })

    if not os.path.exists(args.gaf):
        return

    valid_genes = load_valid_genes(args.genes)
    if not valid_genes:
        return

    all_gene_symbols = parse_gaf_file(args.gaf, valid_genes)
    if not all_gene_symbols:
        return

    max_signature_size = max((info['size'] for info in signatures_info), default=0)
    pool_size = max(args.pool_size, args.steps * max_signature_size)

    modes_to_run = ['cumulative', 'fixed'] if args.mode == 'both' else [args.mode]

    for info in tqdm(signatures_info, desc="Signatures", unit="sig"):
        genes_to_add = info['size']

        for current_mode in modes_to_run:
            random_pool = create_random_pool(all_gene_symbols, info['signature'], pool_size)

            dilution_signatures = create_dilution_signatures(
                info['signature'],
                random_pool,
                args.steps,
                genes_to_add,
                current_mode
            )

            save_signatures_for_signature(
                info['name'],
                dilution_signatures,
                args.output,
                current_mode
            )

            info[f'dilution_signatures_{current_mode}'] = dilution_signatures

        info['steps'] = args.steps
        info['genes_per_step'] = genes_to_add


if __name__ == "__main__":
    main()