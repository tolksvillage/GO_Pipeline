import json
import re
from pathlib import Path
from goatools import obo_parser
from tqdm import tqdm
import os
from go_pipeline.scripts.helper.pipeline_state import PipelineState

def load_go_dag(obo_file_path):
    """Load the GO DAG from an OBO file."""
    try:
        go_dag = obo_parser.GODag(obo_file_path)
        return go_dag
    except Exception as e:
        print(f"Error loading GO DAG: {e}")
        return None


def get_go_definition(file_path, go_id):
    """Extract the definition of a GO term from the OBO file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    terms = re.split(r"\[Term\]", content)[1:]
    for term in terms:
        if f"id: {go_id}" in term:
            def_match = re.search(r'def: "([^"]*)"', term)
            if def_match:
                return def_match.group(1)
            return "No definition found"

    return f"{go_id} not found"


def find_genes_for_go_term(target_go_id, go_term_to_genes, go_dag):
    """Find all genes for a GO term, including genes inherited from descendant terms."""
    direct_genes = set(go_term_to_genes.get(target_go_id, []))
    descendant_go_terms = set()

    if target_go_id in go_dag:
        try:
            descendant_go_terms = go_dag[target_go_id].get_all_children()
        except Exception:
            pass

    inherited_genes = set()
    for desc_go_id in descendant_go_terms:
        inherited_genes.update(go_term_to_genes.get(desc_go_id, []))

    total_genes = direct_genes.union(inherited_genes)
    return list(total_genes)


def parse_gene_mapping_file(mapping_file_path):
    """Parse a mapping file from GO terms to genes."""
    go_term_to_genes = {}

    with open(mapping_file_path, "r", encoding="utf-8") as f:
        current_gene = None

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.endswith(":"):
                current_gene = line[:-1]

            elif line.startswith("- ") and current_gene:
                go_match = re.search(r"\(GO:\d+\)", line)
                if go_match:
                    go_id = go_match.group(0)[1:-1]
                    if go_id not in go_term_to_genes:
                        go_term_to_genes[go_id] = []
                    go_term_to_genes[go_id].append(current_gene)

    return go_term_to_genes


def enrich_ranking_json(json_path, obo_file_path, go_dag, mapping_file_path):
    """Enrich a ranking.json or representatives.json file with GO definitions and genes."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    go_term_to_genes = parse_gene_mapping_file(mapping_file_path)

    key = "ranking" if "ranking" in data else "representatives"

    for entry in data.get(key, []):
        go_id = entry.get("go_id")
        if go_id:
            entry["definition"] = get_go_definition(obo_file_path, go_id)
            entry["gene_symbols"] = sorted(
                find_genes_for_go_term(go_id, go_term_to_genes, go_dag)
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def enrich_manifold_analysis_json(json_path, obo_file_path, go_dag, mapping_file_path):
    """Enrich a manifold_analysis_{ontology}.json file with GO definitions and genes."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    go_term_to_genes = parse_gene_mapping_file(mapping_file_path)

    for group in data.get("groups", []):
        if "representative" in group and "go_id" in group["representative"]:
            go_id = group["representative"]["go_id"]
            group["representative"]["definition"] = get_go_definition(obo_file_path, go_id)
            group["representative"]["gene_symbols"] = sorted(
                find_genes_for_go_term(go_id, go_term_to_genes, go_dag)
            )

        for term in group.get("terms", []):
            go_id = term.get("go_id")
            if go_id:
                term["definition"] = get_go_definition(obo_file_path, go_id)
                term["gene_symbols"] = sorted(
                    find_genes_for_go_term(go_id, go_term_to_genes, go_dag)
                )

        for connection in group.get("connections_to_other_groups", []):
            if "source_term" in connection and "go_id" in connection["source_term"]:
                go_id = connection["source_term"]["go_id"]
                connection["source_term"]["definition"] = get_go_definition(obo_file_path, go_id)
                connection["source_term"]["gene_symbols"] = sorted(
                    find_genes_for_go_term(go_id, go_term_to_genes, go_dag)
                )

            if "target_term" in connection and "go_id" in connection["target_term"]:
                go_id = connection["target_term"]["go_id"]
                connection["target_term"]["definition"] = get_go_definition(obo_file_path, go_id)
                connection["target_term"]["gene_symbols"] = sorted(
                    find_genes_for_go_term(go_id, go_term_to_genes, go_dag)
                )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def process_signatures(input_data_path, obo_file_path, state_file=None):
    """Find and enrich all relevant JSON files."""
    go_dag = load_go_dag(obo_file_path)
    if go_dag is None:
        return

    state = PipelineState(state_file or os.path.join(input_data_path, ".pipeline_state.json"))

    files_found = False
    input_path = Path(input_data_path)

    if not input_path.exists():
        print(f"Path does not exist: {input_data_path}")
        return

    signature_dirs = [
        d for d in input_path.iterdir()
        if d.is_dir()
    ]

    pbar = tqdm(signature_dirs, desc="Processing signatures")

    for signature_dir in pbar:
        pbar.set_description(f"{signature_dir.name}")

        ranking_dir = signature_dir / "Dilutions" / "fixed"

        if ranking_dir.exists():
            for json_file in ranking_dir.glob("*.json"):
                if "ranking" in json_file.name or "representatives" in json_file.name:

                    work_key = str(json_file)
                    if state.is_done("go_definitions_ranking", work_key):
                        files_found = True
                        continue

                    mapping_file = (
                        signature_dir
                        / "BP"
                        / "mapping_genes_to_bp"
                        / "map_genes_to_bp.txt"
                    )

                    if mapping_file.exists():
                        try:
                            enrich_ranking_json(
                                json_file,
                                obo_file_path,
                                go_dag,
                                mapping_file
                            )
                            state.mark_done("go_definitions_ranking", work_key)
                            files_found = True
                        except Exception as e:
                            state.mark_failed("go_definitions_ranking", work_key, str(e))
                            pbar.write(f"  Fehler bei {json_file}: {e}")
                            continue

        if signature_dir.name.startswith("diluted_"):
            continue

        param_analysis_dir = signature_dir / "parameter_analysis"

        if param_analysis_dir.exists():
            for ontology_dir in param_analysis_dir.iterdir():
                if ontology_dir.is_dir():

                    ontology = ontology_dir.name
                    ontology_lower = ontology.lower()

                    manifold_file = (
                        ontology_dir /
                        f"manifold_analysis_{ontology}.json"
                    )

                    if manifold_file.exists():

                        work_key = str(manifold_file)
                        if state.is_done("go_definitions_manifold", work_key):
                            files_found = True
                            continue

                        mapping_file = (
                            signature_dir
                            / ontology
                            / f"mapping_genes_to_{ontology_lower}"
                            / f"map_genes_to_{ontology_lower}.txt"
                        )

                        if mapping_file.exists():
                            try:
                                enrich_manifold_analysis_json(
                                    manifold_file,
                                    obo_file_path,
                                    go_dag,
                                    mapping_file
                                )
                                state.mark_done("go_definitions_manifold", work_key)
                                files_found = True
                            except Exception as e:
                                state.mark_failed("go_definitions_manifold", work_key, str(e))
                                pbar.write(f"  Fehler bei {manifold_file}: {e}")
                                continue

    if not files_found:
        print("No JSON files found to enrich.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Enrich ranking, representatives, and manifold analysis JSON files "
            "with GO definitions and gene symbols"
        )
    )
    parser.add_argument(
        "--input_data",
        required=True,
        help="Path to the signature directories",
    )
    parser.add_argument(
        "--obo_file",
        default="data/go-basic.obo",
        help="Path to the GO OBO file",
    )
    parser.add_argument(
        "--state_file",
        default=None,
        help="Pfad zur gemeinsamen Pipeline-Status-Datei (Resume)",
    )
    args = parser.parse_args()

    process_signatures(args.input_data, args.obo_file, state_file=args.state_file)