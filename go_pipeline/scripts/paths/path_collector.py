import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from goatools import obo_parser


NAMESPACE_CONFIG = {
    "BP": {
        "root_term": "GO:0008150",
        "mapping_path": "BP/mapping_genes_to_bp/map_genes_to_bp.txt",
        "output_file": "bp_complete_paths.json",
    },
    "MF": {
        "root_term": "GO:0003674",
        "mapping_path": "MF/mapping_genes_to_mf/map_genes_to_mf.txt",
        "output_file": "mf_complete_paths.json",
    },
    "CC": {
        "root_term": "GO:0005575",
        "mapping_path": "CC/mapping_genes_to_cc/map_genes_to_cc.txt",
        "output_file": "cc_complete_paths.json",
    },
}


def load_ic_data(ic_file: Path) -> dict:
    """Load precomputed IC values from a JSON file."""
    with open(ic_file, "r", encoding="utf-8") as handle:
        return json.load(handle)



def get_term_ic(go_id: str, ic_data: dict) -> float:
    """Return the normalized IC value for a GO term."""
    return ic_data.get(go_id, {}).get("ic_normalized", 0.0)



def parse_gene_mapping(file_path: str) -> dict:
    """
    Parse a gene-to-GO mapping file.

    Expected format:
        GENE_SYMBOL:
        - term name (GO:0000000)
        - another term (GO:0000001)
    """
    go_to_genes = defaultdict(set)

    with open(file_path, "r", encoding="utf-8") as handle:
        current_gene = None

        for line in handle:
            line = line.strip()
            if not line:
                continue

            if line.endswith(":") and not line.startswith("- "):
                current_gene = line[:-1]
            elif line.startswith("- ") and "(GO:" in line and current_gene:
                go_start = line.rfind("(GO:") + 1
                go_end = line.rfind(")")
                if go_start > 0 and go_end > go_start:
                    go_id = line[go_start:go_end]
                    go_to_genes[go_id].add(current_gene)

    return go_to_genes



def get_all_descendants(go_id: str, go_dag) -> set:
    """Return all descendant GO terms for a given GO term."""
    if go_id not in go_dag:
        return set()

    descendants = set()
    stack = [go_id]

    while stack:
        current = stack.pop()
        if current not in go_dag:
            continue

        for child in go_dag[current].children:
            if child.id not in descendants:
                descendants.add(child.id)
                stack.append(child.id)

    return descendants



def get_inherited_genes(go_id: str, go_to_genes: dict, go_dag) -> set:
    """
    Return genes inherited from descendant terms only.

    Direct genes attached to the current term are not included here.
    """
    inherited_genes = set()
    descendants = get_all_descendants(go_id, go_dag)

    for descendant_id in descendants:
        inherited_genes.update(go_to_genes.get(descendant_id, set()))

    return inherited_genes



def get_term_data(go_id: str, go_dag, go_to_genes: dict, namespace: str, ic_data: dict):
    """Collect summary data for a GO term."""
    if go_id not in go_dag:
        return None

    term = go_dag[go_id]
    ic_value = get_term_ic(go_id, ic_data)

    direct_genes = go_to_genes.get(go_id, set())
    inherited_genes = get_inherited_genes(go_id, go_to_genes, go_dag)
    total_genes = direct_genes.union(inherited_genes)

    return {
        "ic": ic_value,
        "genes_direct_count": len(direct_genes),
        "go_annotation": term.name,
        "genes_inherited_count": len(inherited_genes),
        "genes_total_count": len(total_genes),
    }



def get_valid_children(go_id: str, go_dag, go_to_genes: dict, namespace: str, ic_data: dict) -> list:
    """Return child terms with at least one direct or inherited gene."""
    if go_id not in go_dag:
        return []

    valid_children = []
    for child in go_dag[go_id].children:
        child_data = get_term_data(child.id, go_dag, go_to_genes, namespace, ic_data)
        if child_data and child_data["genes_total_count"] > 0:
            valid_children.append(child.id)

    return valid_children



def collect_all_paths(start_go_id: str, go_dag, go_to_genes: dict, namespace: str, ic_data: dict) -> list:
    """
    Collect all valid paths from the namespace root term to leaf terms.

    A term is considered valid if it has at least one direct or inherited gene.
    """
    all_paths = []
    path_id = 1

    def dfs(current_path: list, current_go_id: str) -> None:
        nonlocal path_id

        term_data = get_term_data(current_go_id, go_dag, go_to_genes, namespace, ic_data)
        if not term_data or term_data["genes_total_count"] == 0:
            return

        current_item = {current_go_id: term_data}
        new_path = current_path + [current_item]
        valid_children = get_valid_children(current_go_id, go_dag, go_to_genes, namespace, ic_data)

        if not valid_children:
            all_paths.append(
                {
                    "path_id": f"{namespace}{path_id}",
                    "path": new_path,
                }
            )
            path_id += 1
            return

        children_are_leaves = []
        children_with_descendants = []

        for child_id in valid_children:
            child_valid_children = get_valid_children(child_id, go_dag, go_to_genes, namespace, ic_data)
            if not child_valid_children:
                children_are_leaves.append(child_id)
            else:
                children_with_descendants.append(child_id)

        if children_are_leaves and not children_with_descendants:
            leaf_data = []
            for leaf_id in children_are_leaves:
                leaf_term_data = get_term_data(leaf_id, go_dag, go_to_genes, namespace, ic_data)
                if leaf_term_data:
                    leaf_data.append({leaf_id: leaf_term_data})

            if leaf_data:
                all_paths.append(
                    {
                        "path_id": f"{namespace}{path_id}",
                        "path": new_path + [leaf_data],
                    }
                )
                path_id += 1
            return

        for leaf_id in children_are_leaves:
            leaf_term_data = get_term_data(leaf_id, go_dag, go_to_genes, namespace, ic_data)
            if leaf_term_data:
                all_paths.append(
                    {
                        "path_id": f"{namespace}{path_id}",
                        "path": new_path + [{leaf_id: leaf_term_data}],
                    }
                )
                path_id += 1

        for child_id in children_with_descendants:
            dfs(new_path, child_id)

    dfs([], start_go_id)
    return all_paths



def create_complete_go_term_dictionary(go_dag, go_to_genes: dict, namespace: str, ic_data: dict) -> dict:
    """
    Build a full lookup dictionary for all GO terms in one namespace.

    Each entry contains the term name, direct gene count, inherited gene count,
    total gene count, and normalized IC value.
    """
    namespace_map = {
        "BP": "biological_process",
        "MF": "molecular_function",
        "CC": "cellular_component",
    }

    target_namespace = namespace_map.get(namespace, "biological_process")
    go_term_dict = {}

    for go_id, go_term in go_dag.items():
        if getattr(go_term, "namespace", None) != target_namespace:
            continue

        term_data = get_term_data(go_id, go_dag, go_to_genes, namespace, ic_data)
        if not term_data:
            continue

        go_term_dict[go_id] = {
            "go_annotation": term_data["go_annotation"],
            "genes_direct_count": term_data["genes_direct_count"],
            "genes_inherited_count": term_data["genes_inherited_count"],
            "genes_total_count": term_data["genes_total_count"],
            "ic": term_data["ic"],
        }
    return go_term_dict



def save_complete_go_term_dictionary(dir_path: Path, go_dag, go_to_genes: dict, namespace: str, ic_data: dict) -> bool:
    """Create and save the complete GO term dictionary for one namespace."""
    try:
        go_term_dict = create_complete_go_term_dictionary(go_dag, go_to_genes, namespace, ic_data)
        if not go_term_dict:
            print(f"    {namespace}: no GO terms found")
            return False

        output_dir = dir_path / "collected_paths"
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / f"{namespace.lower()}_complete_terms.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(go_term_dict, handle, indent=2, ensure_ascii=False)
        return True

    except Exception as exc:
        print(f"    {namespace}: failed to save GO term dictionary: {exc}")
        return False



def process_single_namespace(dir_path: Path, go_dag, namespace: str) -> bool:
    """Process one namespace inside one result directory."""
    try:
        config = NAMESPACE_CONFIG[namespace]
        root_term = config["root_term"]
        mapping_file = dir_path / config["mapping_path"]
        output_file = config["output_file"]

        if not mapping_file.exists():
            print(f"    {namespace}: skipped, missing {mapping_file.name}")
            return False

        ic_file = Path("data/GO_IC") / f"{namespace.lower()}_ic.json"
        if not ic_file.exists():
            print(f"    {namespace}: skipped, missing {ic_file.name}")
            return False

        ic_data = load_ic_data(ic_file)
        go_to_genes = parse_gene_mapping(str(mapping_file))

        if not go_to_genes:
            print(f"    {namespace}: no gene mappings found")
            return False

        total_annotations = sum(len(genes) for genes in go_to_genes.values())

        dictionary_success = save_complete_go_term_dictionary(
            dir_path,
            go_dag,
            go_to_genes,
            namespace,
            ic_data,
        )

        if root_term not in go_dag:
            return dictionary_success

        all_paths = collect_all_paths(root_term, go_dag, go_to_genes, namespace, ic_data)

        if not all_paths:
            print(f"    {namespace}: no paths found")
            return dictionary_success

        output_dir = dir_path / "collected_paths"
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / output_file
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(all_paths, handle, indent=2, ensure_ascii=False)

        return True

    except Exception as exc:
        print(f"    {namespace}: failed: {exc}")
        return False



def process_single_directory(dir_path: Path, go_dag, namespaces=None) -> dict:
    """Process all requested namespaces for one directory."""
    if namespaces is None:
        namespaces = ["BP", "MF", "CC"]

    results = {}

    for namespace in namespaces:
        results[namespace] = process_single_namespace(dir_path, go_dag, namespace)

    successful = sum(1 for success in results.values() if success)
    return results



def process_all_directories(base_dir: str, obo_file: str, namespaces=None) -> None:
    """Process all result directories under the given base directory."""
    if namespaces is None:
        namespaces = ["BP", "MF", "CC"]

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory does not exist: {base_dir}")
        return

    try:
        go_dag = obo_parser.GODag(obo_file, optional_attrs=["relationship"])
    except Exception as exc:
        print(f"Failed to load GO DAG: {exc}")
        return

    subdirs = [
        path
        for path in base_path.iterdir()
        if path.is_dir() and not path.name.startswith("diluted_")
    ]

    total_successful = 0
    total_failed = 0
    namespace_stats = {ns: {"successful": 0, "failed": 0} for ns in namespaces}

    for dir_path in tqdm(subdirs, desc="Signatures", unit="sig"):
        results = process_single_directory(dir_path, go_dag, namespaces)

        if any(results.values()):
            total_successful += 1
        else:
            total_failed += 1

        for namespace, success in results.items():
            if success:
                namespace_stats[namespace]["successful"] += 1
            else:
                namespace_stats[namespace]["failed"] += 1

    for namespace in namespaces:
        stats = namespace_stats[namespace]


    if total_successful > 0:
        for namespace in namespaces:
            config = NAMESPACE_CONFIG[namespace]




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect GO paths for multiple namespaces across result directories"
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Base directory containing result subdirectories",
    )
    parser.add_argument(
        "--obo_file",
        default="data/go-basic.obo",
        help="Path to the GO OBO file (default: data/go-basic.obo)",
    )
    parser.add_argument(
        "--namespaces",
        nargs="+",
        default=["BP", "MF", "CC"],
        choices=["BP", "MF", "CC"],
        help="Namespaces to process (default: BP MF CC)",
    )

    args = parser.parse_args()

    if not Path(args.obo_file).exists():
        print(f"OBO file not found: {args.obo_file}")
        sys.exit(1)

    for namespace in args.namespaces:
        if namespace not in NAMESPACE_CONFIG:
            sys.exit(1)

    process_all_directories(args.base_dir, args.obo_file, args.namespaces)


if __name__ == "__main__":
    main()
