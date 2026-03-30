import os
import argparse
import contextlib
import io
from pathlib import Path
from tqdm import tqdm
from goatools.anno.gaf_reader import GafReader
from go_pipeline.init_GO import initialize_go


@contextlib.contextmanager
def suppress_output():
    """
    Suppresses stdout and stderr temporarily.
    Useful to keep terminal output minimal.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def get_most_specific_terms(gene_go_dict, go_dag):
    """
    Removes redundant parent terms and keeps only the most specific annotations.
    """
    refined_dict = {}

    for gene, go_terms in gene_go_dict.items():
        if not go_terms:
            refined_dict[gene] = {}
            continue

        go_ids = set(go_terms.keys())
        most_specific = set(go_ids)

        for go_id in go_ids:
            if go_id not in go_dag:
                continue

            ancestors = go_dag[go_id].get_all_parents()
            most_specific -= ancestors

        refined_dict[gene] = {
            go_id: desc for go_id, desc in go_terms.items()
            if go_id in most_specific
        }

    return refined_dict


def getgoids_cached(output_path=None, data_file=None, output_mapping=None,
                    namespace=None, go_dag=None, annotations=None):
    """
    Removes redundant parent terms and keeps only the most specific annotations.
    """
    namespace_mapping = {
        'BP': 'biological_process',
        'MF': 'molecular_function',
        'CC': 'cellular_component'
    }

    if namespace not in namespace_mapping:
        raise ValueError(f"Invalid namespace: {namespace}. Allowed values: BP, MF, CC")

    target_namespace = namespace_mapping[namespace]

    try:
        with open(data_file) as df:
            gene_list = [line.strip() for line in df.readlines() if line.strip()]
    except FileNotFoundError:
        return set(), 0

    target_genes = set(gene_list)
    num_genes = len(gene_list)

    gene_go_dict = {}
    my_terms = set()
    processed_genes = set()

    for gene in gene_list:
        gene_go_dict[gene] = {}

    for annotation in annotations:
        gene_symbol = annotation.DB_Symbol

        if gene_symbol not in target_genes:
            continue

        processed_genes.add(gene_symbol)

        go_id = annotation.GO_ID

        if go_id not in go_dag:
            continue

        go_term = go_dag[go_id]

        if go_term.namespace == target_namespace:
            description = go_term.name
            gene_go_dict[gene_symbol][go_id] = description
            my_terms.add(go_id)

    genes_without_annotations = target_genes - processed_genes
    for gene in genes_without_annotations:
        gene_go_dict[gene] = {}

    gene_go_dict = get_most_specific_terms(gene_go_dict, go_dag)

    my_terms = set()
    for go_terms in gene_go_dict.values():
        my_terms.update(go_terms.keys())

    if output_mapping:
        os.makedirs(output_mapping, exist_ok=True)
        mapping_file = os.path.join(output_mapping, f'map_genes_to_{namespace.lower()}.txt')

        with open(mapping_file, "w", encoding="utf-8") as f:
            for gene_id in gene_list:
                go_terms = gene_go_dict[gene_id]
                f.write(f"{gene_id}:\n")

                if go_terms:
                    for go_id, desc in sorted(go_terms.items(), key=lambda x: x[1]):
                        go_id_formatted = go_id.split(':')[1] if ':' in go_id else go_id
                        f.write(f"- {desc} (GO:{go_id_formatted})\n")
                else:
                    f.write(f"- No {namespace} annotations found\n")
                f.write("\n")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for term in sorted(my_terms):
                f.write(f"{term}\n")

    return my_terms, num_genes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--namespaces", nargs='+', choices=['BP', 'MF', 'CC'], default=['BP', 'MF', 'CC'])
    args = parser.parse_args()

    base_path = Path(args.base_path)
    signature_files = sorted(base_path.glob("*.txt"))

    if not signature_files:
        return

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    gaf_path = PROJECT_ROOT / "data" / "goa_human.gaf"

    with suppress_output():
        gaf_reader = GafReader(str(gaf_path))
        all_annotations = gaf_reader.associations

    go_dags = {}
    for namespace in args.namespaces:
        with suppress_output():
            godag, _, _, _, _ = initialize_go(namespace)
        go_dags[namespace] = godag

    with tqdm(signature_files, desc="Signatures", unit="sig") as pbar:
        for signature_file in pbar:
            signature_name = signature_file.stem
            signature_output = Path(args.output_path) / signature_name

            for namespace in args.namespaces:
                try:
                    namespace_path = signature_output / namespace
                    namespace_path.mkdir(parents=True, exist_ok=True)

                    getgoids_cached(
                        output_path=str(namespace_path / f'my_terms_{namespace}.txt'),
                        data_file=str(signature_file),
                        output_mapping=str(namespace_path / f'mapping_genes_to_{namespace.lower()}'),
                        namespace=namespace,
                        go_dag=go_dags[namespace],
                        annotations=all_annotations
                    )

                except Exception:
                    pass


if __name__ == "__main__":
    main()