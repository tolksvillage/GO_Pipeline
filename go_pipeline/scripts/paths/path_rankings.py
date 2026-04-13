import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import sys

sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")

def load_json(filepath):
    """Load a JSON file."""
    with open(filepath, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_gene_signature(filepath):
    """Load a gene signature file and return the number of genes."""
    with open(filepath, "r", encoding="utf-8") as handle:
        genes = [line.strip() for line in handle if line.strip()]
    return len(genes)


def find_term_in_path(path, target_go_id):
    """Return the index of a GO term in a path, or None if not found."""
    for idx, node in enumerate(path):
        if isinstance(node, list):
            for subnode in node:
                if target_go_id in subnode:
                    return idx
        elif isinstance(node, dict):
            if target_go_id in node:
                return idx
    return None


def crop_path_to_term(path, target_go_id):
    """Trim a path so that it ends at the target GO term."""
    term_idx = find_term_in_path(path, target_go_id)
    if term_idx is None:
        return None
    return path[: term_idx + 1]


def extract_node_data(node):
    """Extract (go_id, data) pairs from a node."""
    results = []
    if isinstance(node, list):
        for subnode in node:
            for go_id, data in subnode.items():
                results.append((go_id, data))
    elif isinstance(node, dict):
        for go_id, data in node.items():
            results.append((go_id, data))
    return results


def calculate_node_metric(node_data, ic_raw, n_signature, alpha, beta):
    """
    Compute the metric for a single node.

    metric = ((alpha * ln(genes_direct_count) + beta * ln(genes_inherited_count))
              * ic_normalized) / expectation

    expectation = max(1, n_signature * exp(-ic_raw))
    """
    genes_direct = node_data["genes_direct_count"]
    genes_inherited = node_data["genes_inherited_count"]
    ic_normalized = node_data["ic"]

    expectation = max(1.0, n_signature * math.exp(-ic_raw))

    if genes_direct <= 0 and genes_inherited <= 0:
        return 0.0

    direct_term = alpha * math.log(genes_direct) if genes_direct > 0 else 0.0
    inherited_term = beta * math.log(genes_inherited) if genes_inherited > 0 else 0.0

    numerator = (direct_term + inherited_term) * ic_normalized
    metric = numerator / expectation
    return metric


def calculate_path_score(cropped_path, ic_data, n_signature, alpha, beta):
    """Compute the average score for a cropped path."""
    total_metric = 0.0
    node_count = 0

    for node in cropped_path:
        node_entries = extract_node_data(node)

        for go_id, node_data in node_entries:
            if go_id not in ic_data:
                continue

            ic_raw = ic_data[go_id]["ic_raw"]
            metric = calculate_node_metric(node_data, ic_raw, n_signature, alpha, beta)
            total_metric += metric
            node_count += 1

    if node_count == 0:
        return 0.0

    return total_metric / node_count


def extract_go_ids_from_path(path):
    """Extract GO IDs from a path in order."""
    go_ids = []
    for node in path:
        if isinstance(node, list):
            for subnode in node:
                for go_id in subnode.keys():
                    go_ids.append(go_id)
        elif isinstance(node, dict):
            for go_id in node.keys():
                go_ids.append(go_id)
    return go_ids


def extract_children_terms_details(complete_path, target_go_id, path_id):
    """
    Extract all descendant terms after the target term, including metadata.
    """
    term_idx = find_term_in_path(complete_path, target_go_id)
    if term_idx is None or term_idx >= len(complete_path) - 1:
        return []

    children_details = []

    for node in complete_path[term_idx + 1 :]:
        if isinstance(node, list):
            for subnode in node:
                for go_id, data in subnode.items():
                    children_details.append(
                        {
                            "go_id": go_id,
                            "go_annotation": data.get("go_annotation", ""),
                            "ic": data.get("ic", 0.0),
                            "genes_direct_count": data.get("genes_direct_count", 0),
                            "genes_inherited_count": data.get("genes_inherited_count", 0),
                            "genes_total_count": data.get("genes_total_count", 0),
                            "path_id": path_id,
                        }
                    )
        elif isinstance(node, dict):
            for go_id, data in node.items():
                children_details.append(
                    {
                        "go_id": go_id,
                        "go_annotation": data.get("go_annotation", ""),
                        "ic": data.get("ic", 0.0),
                        "genes_direct_count": data.get("genes_direct_count", 0),
                        "genes_inherited_count": data.get("genes_inherited_count", 0),
                        "genes_total_count": data.get("genes_total_count", 0),
                        "path_id": path_id,
                    }
                )

    return children_details


def find_unique_cropped_paths_for_go_id(go_id, complete_paths):
    """
    Find all unique cropped paths for a GO term and collect child term details.
    """
    unique_paths_dict = {}
    all_children = {}

    for path_entry in complete_paths:
        path_id = path_entry["path_id"]
        path = path_entry["path"]

        if find_term_in_path(path, go_id) is None:
            continue

        cropped_path = crop_path_to_term(path, go_id)
        if cropped_path is None:
            continue

        go_ids_in_path = extract_go_ids_from_path(cropped_path)
        path_fingerprint = "|".join(go_ids_in_path)

        children_details = extract_children_terms_details(path, go_id, path_id)

        for child in children_details:
            child_go_id = child["go_id"]
            if child_go_id not in all_children:
                all_children[child_go_id] = {
                    "go_id": child_go_id,
                    "go_annotation": child["go_annotation"],
                    "ic": child["ic"],
                    "genes_direct_count": child["genes_direct_count"],
                    "genes_inherited_count": child["genes_inherited_count"],
                    "genes_total_count": child["genes_total_count"],
                    "source_path_ids": [path_id],
                }
            else:
                if path_id not in all_children[child_go_id]["source_path_ids"]:
                    all_children[child_go_id]["source_path_ids"].append(path_id)

        if path_fingerprint not in unique_paths_dict:
            unique_paths_dict[path_fingerprint] = {
                "cropped_path": cropped_path,
                "source_path_ids": [path_id],
            }
        else:
            unique_paths_dict[path_fingerprint]["source_path_ids"].append(path_id)

    unique_paths = list(unique_paths_dict.values())
    children_terms = list(all_children.values())

    return unique_paths, children_terms


def analyze_term_paths(term, complete_paths, ic_data, n_signature):
    """
    Analyze all paths for one GO term across all parameter combinations.
    """
    target_go_id = term["go_id"]

    unique_paths, children_terms = find_unique_cropped_paths_for_go_id(
        target_go_id, complete_paths
    )

    if not unique_paths:
        return {
            "go_id": target_go_id,
            "name": term["name"],
            "total_unique_paths": 0,
            "total_source_paths": 0,
            "unique_paths_info": [],
            "children_terms": [],
            "rankings": {},
        }

    total_source_paths = sum(len(path["source_path_ids"]) for path in unique_paths)

    alpha_values = [round(x * 0.1, 1) for x in range(0, 11)]
    beta_values = [round(x * 0.1, 1) for x in range(0, 11)]

    configurations = []
    for alpha in alpha_values:
        for beta in beta_values:
            if alpha == 0 and beta == 0:
                continue
            configurations.append((alpha, beta))

    path_rankings = defaultdict(list)
    path_metrics = defaultdict(list)

    for alpha, beta in configurations:
        path_scores = {}

        for idx, unique_path_info in enumerate(unique_paths):
            cropped_path = unique_path_info["cropped_path"]
            score = calculate_path_score(cropped_path, ic_data, n_signature, alpha, beta)
            path_scores[idx] = score

        sorted_paths = sorted(path_scores.items(), key=lambda item: item[1], reverse=True)

        for rank, (path_idx, score) in enumerate(sorted_paths, start=1):
            path_rankings[path_idx].append(rank)
            path_metrics[path_idx].append(float(score))

    final_rankings = {}
    for path_idx in path_rankings.keys():
        rankings = path_rankings[path_idx]
        metrics = path_metrics[path_idx]

        mean_rank = np.mean(rankings)
        median_rank = np.median(rankings)

        metrics_by_config = {}
        for idx, (alpha, beta) in enumerate(configurations):
            config_key = f"alpha_{alpha}_beta_{beta}"
            metrics_by_config[config_key] = {
                "alpha": alpha,
                "beta": beta,
                "metric": float(metrics[idx]),
                "rank": rankings[idx],
            }

        final_rankings[path_idx] = {
            "mean_rank": float(mean_rank),
            "median_rank": float(median_rank),
            "all_ranks": rankings,
            "all_metrics": metrics_by_config,
        }

    sorted_final = sorted(final_rankings.items(), key=lambda item: item[1]["mean_rank"])

    unique_paths_info = []
    rankings_dict = {}

    for rank_position, (path_idx, ranking_data) in enumerate(sorted_final, start=1):
        unique_path_id = f"unique_{path_idx + 1}"

        unique_paths_info.append(
            {
                "unique_path_id": unique_path_id,
                "rank_position": rank_position,
                "cropped_path": unique_paths[path_idx]["cropped_path"],
                "source_path_ids": unique_paths[path_idx]["source_path_ids"],
            }
        )

        rankings_dict[unique_path_id] = ranking_data

    return {
        "go_id": target_go_id,
        "name": term["name"],
        "total_unique_paths": len(unique_paths),
        "total_source_paths": total_source_paths,
        "unique_paths_info": unique_paths_info,
        "children_terms": children_terms,
        "rankings": rankings_dict,
    }


def extract_go_ids_from_cropped_path(cropped_path):
    """Extract all GO IDs from a cropped path."""
    go_ids = set()
    for node in cropped_path:
        if isinstance(node, list):
            for subnode in node:
                for go_id in subnode.keys():
                    go_ids.add(go_id)
        elif isinstance(node, dict):
            for go_id in node.keys():
                go_ids.add(go_id)
    return go_ids


def visualize_term(term_data, godag, output_file, go_to_rs, best_path_go_ids):
    """
    Render a GO term with its unique paths and child terms as a PNG.
    """
    try:
        from goatools.gosubdag.gosubdag import GoSubDag
        from goatools.gosubdag.plot.go2color import Go2Color
        from goatools.gosubdag.plot.gosubdag_plot import GoSubDagPlot

        target_go_id = term_data["go_id"]
        term_name = term_data["name"]

        path_go_ids = set()
        go_to_data = {}

        for unique_path_info in term_data.get("unique_paths_info", []):
            cropped_path = unique_path_info["cropped_path"]
            for node in cropped_path:
                if isinstance(node, list):
                    for subnode in node:
                        for go_id, data in subnode.items():
                            path_go_ids.add(go_id)
                            go_to_data[go_id] = data
                elif isinstance(node, dict):
                    for go_id, data in node.items():
                        path_go_ids.add(go_id)
                        go_to_data[go_id] = data

        children_go_ids = set()
        for child in term_data.get("children_terms", []):
            child_go_id = child["go_id"]
            children_go_ids.add(child_go_id)
            go_to_data[child_go_id] = {
                "go_annotation": child["go_annotation"],
                "ic": child["ic"],
                "genes_direct_count": child["genes_direct_count"],
                "genes_inherited_count": child["genes_inherited_count"],
                "genes_total_count": child["genes_total_count"],
            }

        all_go_ids = path_go_ids | children_go_ids
        if not all_go_ids:
            return False

        gosubdag = GoSubDag(list(all_go_ids), godag, relationships=None, prt=None)

        go2color = {}
        go2label = {}
        go2attr = {}

        for goid in gosubdag.go2obj.keys():
            go_obj = gosubdag.go2obj[goid]
            label_parts = [go_obj.name]

            if goid in go_to_data:
                data = go_to_data[goid]
                ic = data.get("ic", 0.0)
                genes_d = data.get("genes_direct_count", 0)
                genes_i = data.get("genes_inherited_count", 0)
                genes_t = data.get("genes_total_count", 0)

                label_parts.append(f"IC: {ic:.3f}")
                label_parts.append(f"Genes (D|I|T): {genes_d}|{genes_i}|{genes_t}")

                if goid in go_to_rs:
                    rs = go_to_rs[goid]
                    label_parts.append(f"RS: {rs:.1f}%")

            go2label[goid] = "\n".join(label_parts)

            if goid == target_go_id:
                go2color[goid] = "#DDA0DD"
            elif goid in path_go_ids and goid != target_go_id:
                go2color[goid] = "#90EE90"
            elif goid in children_go_ids:
                go2color[goid] = "#87CEEB"
            else:
                go2color[goid] = "#FFFFFF"

            if goid in best_path_go_ids:
                go2attr[goid] = {"penwidth": "3.0"}
            else:
                go2attr[goid] = {"penwidth": "1.0"}

        objcolor = Go2Color(gosubdag, go2color=go2color)

        plot_kwargs = {
            "Go2Color": objcolor,
            "rankdir": "TB",
            "dpi": 150,
            "title": f"{term_name}\n({target_go_id})",
            "id2label": go2label,
            "id2attr": go2attr,
        }

        plot_obj = GoSubDagPlot(gosubdag, **plot_kwargs)
        plot_obj.plt_dag(str(output_file))

        return True

    except Exception:
        return False


def create_all_visualizations(rankings_data, godag, vis_dir, ontology, manifold_data):
    """Create PNG visualizations for all ranked terms."""
    vis_dir.mkdir(parents=True, exist_ok=True)

    go_to_rs = {}
    for group in manifold_data.get("groups", []):
        for term in group.get("terms", []):
            go_to_rs[term["go_id"]] = term.get("frequency_percentage", 0.0)

    for result in rankings_data["results"]:
        term_name = result["name"]

        best_path_go_ids = set()
        for unique_path_info in result.get("unique_paths_info", []):
            if unique_path_info.get("rank_position") == 1:
                cropped_path = unique_path_info["cropped_path"]
                for node in cropped_path:
                    if isinstance(node, list):
                        for subnode in node:
                            for go_id_node in subnode.keys():
                                best_path_go_ids.add(go_id_node)
                    elif isinstance(node, dict):
                        for go_id_node in node.keys():
                            best_path_go_ids.add(go_id_node)
                break

        safe_name = re.sub(r"[^\w\s-]", "", term_name).strip().replace(" ", "_")
        safe_name = re.sub(r"[-\s]+", "_", safe_name).lower()
        output_file = vis_dir / f"{safe_name}.png"

        visualize_term(result, godag, output_file, go_to_rs, best_path_go_ids)


def enrich_manifold_with_rankings(manifold_file, rankings_data):
    """
    Update a manifold analysis JSON file with path ranking information.
    """
    manifold_data = load_json(manifold_file)

    go_id_to_data = {}
    for result in rankings_data["results"]:
        go_id = result["go_id"]
        go_id_to_data[go_id] = {
            "unique_paths_info": result["unique_paths_info"],
            "rankings": result["rankings"],
            "total_unique_paths": result["total_unique_paths"],
            "total_source_paths": result["total_source_paths"],
            "children_terms": result["children_terms"],
        }

    solution_path_ids = set()

    for group in manifold_data.get("groups", []):
        for term in group.get("terms", []):
            go_id = term["go_id"]

            if go_id not in go_id_to_data:
                continue

            term_data = go_id_to_data[go_id]
            ranked_unique_paths = []

            for unique_path_info in term_data["unique_paths_info"]:
                unique_path_id = unique_path_info["unique_path_id"]
                ranking_data = term_data["rankings"][unique_path_id]

                for source_path_id in unique_path_info["source_path_ids"]:
                    solution_path_ids.add(source_path_id)

                ranked_unique_paths.append(
                    {
                        unique_path_id: {
                            "rank_position": unique_path_info["rank_position"],
                            "mean_rank": ranking_data["mean_rank"],
                            "median_rank": ranking_data["median_rank"],
                            "cropped_path": unique_path_info["cropped_path"],
                            "source_path_ids": unique_path_info["source_path_ids"],
                        }
                    }
                )

            term["path_ids"] = ranked_unique_paths
            term["total_unique_paths"] = term_data["total_unique_paths"]
            term["total_source_paths"] = term_data["total_source_paths"]
            term["children_terms"] = term_data["children_terms"]

    with open(manifold_file, "w", encoding="utf-8") as handle:
        json.dump(manifold_data, handle, indent=2, ensure_ascii=False)

    return manifold_data, solution_path_ids


def create_manifold_complete_paths(complete_paths, solution_path_ids, output_file):
    """
    Save only the complete paths that are used by the selected solution set.
    """
    manifold_paths = []
    for path_entry in complete_paths:
        if path_entry["path_id"] in solution_path_ids:
            manifold_paths.append(path_entry)

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(manifold_paths, handle, indent=2, ensure_ascii=False)

    return manifold_paths


def process_signature_ontology(
    signature_name,
    ontology,
    results_dir,
    signatures_dir,
    ic_data_dir,
    obo_file,
):
    """Process one signature for one ontology."""
    signature_result_dir = results_dir / signature_name

    manifold_file = (
        signature_result_dir
        / "parameter_analysis"
        / ontology.upper()
        / f"manifold_analysis_{ontology.upper()}.json"
    )
    complete_paths_file = (
        signature_result_dir / "collected_paths" / f"{ontology}_complete_paths.json"
    )
    signature_file = signatures_dir / f"{signature_name}.txt"
    ic_file = ic_data_dir / f"{ontology}_ic.json"

    required_files = [
        manifold_file,
        complete_paths_file,
        signature_file,
        ic_file,
        obo_file,
    ]

    if not all(path.exists() for path in required_files):
        return None

    try:
        from goatools.obo_parser import GODag

        complete_paths = load_json(complete_paths_file)
        manifold_data = load_json(manifold_file)
        ic_data = load_json(ic_file)
        n_signature = load_gene_signature(signature_file)
        godag = GODag(str(obo_file), optional_attrs=[], prt=None)
    except Exception:
        return None

    all_terms = []
    for group in manifold_data.get("groups", []):
        for term in group.get("terms", []):
            all_terms.append(term)

    if not all_terms:
        return None

    results = []
    for term in all_terms:
        term_result = analyze_term_paths(term, complete_paths, ic_data, n_signature)
        results.append(term_result)

    output_data = {
        "metadata": {
            "signature_name": signature_name,
            "ontology": ontology.upper(),
            "n_signature": n_signature,
            "parameter_configurations": 120,
        },
        "results": results,
    }

    output_dir = signature_result_dir / "path_rankings"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"term_path_rankings_{ontology.upper()}.json"
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2, ensure_ascii=False)

    manifold_data_enriched, solution_path_ids = enrich_manifold_with_rankings(
        manifold_file, output_data
    )

    manifold_paths_file = output_dir / f"manifold_complete_paths_{ontology.upper()}.json"
    create_manifold_complete_paths(complete_paths, solution_path_ids, manifold_paths_file)

    vis_dir = output_dir / "visualizations" / ontology.upper()
    create_all_visualizations(output_data, godag, vis_dir, ontology, manifold_data_enriched)

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Path ranking analysis for GO terms across multiple signatures and ontologies"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Results directory, for example: Analysis/test/results",
    )
    parser.add_argument(
        "--ic_data_dir",
        type=str,
        default="data/GO_IC",
        help="Directory containing IC data files",
    )
    parser.add_argument(
        "--obo_file",
        type=str,
        default="data/go-basic.obo",
        help="GO OBO file used for visualization",
    )
    parser.add_argument(
        "--ontologies",
        type=str,
        nargs="+",
        default=["bp", "mf", "cc"],
        help="Ontologies to process",
    )

    args = parser.parse_args()

    results_dir = Path(args.input_dir)
    ic_data_dir = Path(args.ic_data_dir)
    obo_file = Path(args.obo_file)

    base_dir = results_dir.parent
    signatures_dir = base_dir / "signatures"

    if not results_dir.exists():
        sys.exit(1)

    if not signatures_dir.exists():
        sys.exit(1)

    if not ic_data_dir.exists():
        sys.exit(1)

    if not obo_file.exists():
        sys.exit(1)

    signature_dirs = [
        directory
        for directory in results_dir.iterdir()
        if directory.is_dir() and not directory.name.startswith("diluted_")
    ]

    if not signature_dirs:
        sys.exit(1)

    total_processed = 0
    total_failed = 0
    all_output_files = []

    for signature_dir in tqdm(
            signature_dirs,
            desc="Processing signatures",
            unit="signature",
            file=sys.__stdout__
    ):
        signature_name = signature_dir.name

        for ontology in args.ontologies:
            output_file = process_signature_ontology(
                signature_name=signature_name,
                ontology=ontology,
                results_dir=results_dir,
                signatures_dir=signatures_dir,
                ic_data_dir=ic_data_dir,
                obo_file=obo_file,
            )

            if output_file:
                total_processed += 1
                all_output_files.append(output_file)
            else:
                total_failed += 1


if __name__ == "__main__":
    main()