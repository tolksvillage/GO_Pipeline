"""
Initialization of the Gene Ontology data for different namespaces.
Supports flexible selection of BP (Biological Process), MF (Molecular Function), or CC (Cellular Component).
"""

import gzip
import time
import json
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

from goatools.obo_parser import GODag
from goatools.associations import dnld_assc
import Bio.UniProt.GOA as GOA

from go_pipeline.scripts.helper.termcounts import TermCounts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def download_file(url, target_path):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urlopen(req) as response, open(target_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def download_go_data(data_dir=None):
    """
    Downloads the required GO data if it is not available,
    and extracts the GAF file into a normal .gaf file.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    obo_path = data_dir / "go-basic.obo"
    if not obo_path.is_file():
        print("Downloading go-basic.obo...")
        obo_url = "https://current.geneontology.org/ontology/go-basic.obo"
        download_file(obo_url, obo_path)

    gaf_gz_path = data_dir / "goa_human.gaf.gz"
    if not gaf_gz_path.is_file():
        print("Downloading goa_human.gaf.gz...")
        gaf_url = "https://current.geneontology.org/annotations/goa_human.gaf.gz"
        download_file(gaf_url, gaf_gz_path)


    gaf_path = data_dir / "goa_human.gaf"
    if not gaf_path.is_file():
        print("Extracting goa_human.gaf...")
        with gzip.open(gaf_gz_path, "rb") as f_in, open(gaf_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    gene_info_gz_path = data_dir / "Homo_sapiens.gene_info.gz"
    if not gene_info_gz_path.is_file():
        print("Downloading Homo_sapiens.gene_info.gz...")
        gene_info_url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
        download_file(gene_info_url, gene_info_gz_path)

    return obo_path, gaf_path, gene_info_gz_path


def name2terms(gene_names, annotations):
    go_term_set = set()
    for ann in annotations:
        if ann["DB_Object_Symbol"] in gene_names:
            go_term_set.add(ann["GO_ID"])
    return list(sorted(go_term_set))


def name2terms_counts(gene_names, annotations):
    """
    Counts how often each GO term is associated with the given genes.
    """
    go_term_counts = {}
    for ann in annotations:
        if ann["DB_Object_Symbol"] in gene_names:
            go_id = ann["GO_ID"]
            go_term_counts[go_id] = go_term_counts.get(go_id, 0) + 1
    return go_term_counts


def initialize_go(namespace_code="BP", data_dir=None):
    """
    Initializes the GO data for the specified namespace.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    start_time = time.time()

    namespace_mappings = {
        "BP": ("BP", ["P"]),
        "MF": ("MF", ["F"]),
        "CC": ("CC", ["C"]),
    }

    if namespace_code not in namespace_mappings:
        raise ValueError(f"Invalid namespace: {namespace_code}. Allowed values: BP, MF, CC")

    namespace, type_codes = namespace_mappings[namespace_code]

    obo_path, gaf_path, _ = download_go_data(data_dir)

    cache_dir = data_dir / "graph_informations" / "cache_jsons"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_info_file = cache_dir / "cache_info.json"

    rebuild_cache = True
    if cache_info_file.exists():
        try:
            with open(cache_info_file, "r", encoding="utf-8") as f:
                cache_info = json.load(f)
            obo_mtime = obo_path.stat().st_mtime
            if obo_mtime <= cache_info.get("obo_mtime", 0):
                rebuild_cache = False
        except Exception as e:
            print(f"Error reading cache info: {e}")
            rebuild_cache = True

    print("Loading GODag...")
    godag = GODag(str(obo_path), prt=None)

    if rebuild_cache:
        try:
            with open(cache_info_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "obo_mtime": obo_path.stat().st_mtime,
                        "last_loaded": time.time(),
                    },
                    f,
                )
        except Exception as e:
            print(f"Error saving cache info: {e}")

    print(f"Loading Annotations for {namespace}...")
    annotations = []

    with open(gaf_path, "r", encoding="utf-8") as gaf_fp:
        for entry in GOA.gafiterator(gaf_fp):
            if entry["Aspect"] in type_codes and "NOT" not in entry["Qualifier"]:
                annotations.append(entry)

    assoc_cache_file = cache_dir / f"associations_{namespace}.json"

    if assoc_cache_file.exists():
        print(f"Loading {namespace} Associations from cache...")
        try:
            with open(assoc_cache_file, "r", encoding="utf-8") as f:
                assoc_dict = json.load(f)

            associations = {}
            for gene_id, terms in assoc_dict.items():
                associations[gene_id] = set(terms)

        except Exception as e:
            print(f"Error loading associations from cache: {e}")
            associations = dnld_assc(str(gaf_path), godag, namespace=namespace)
    else:
        print(f"Creating new {namespace} Associations...")
        associations = dnld_assc(str(gaf_path), godag, namespace=namespace)

        assoc_dict = {}
        for gene_id, terms in associations.items():
            assoc_dict[gene_id] = list(terms)

        try:
            with open(assoc_cache_file, "w", encoding="utf-8") as f:
                json.dump(assoc_dict, f)
            print(f"{namespace} Associations saved to cache")
        except Exception as e:
            print(f"Error saving {namespace} associations: {e}")

    term_count = TermCounts(godag, associations, annotations)

    end_time = time.time()
    print(f"Initialization of {namespace} completed in {end_time - start_time:.2f} seconds")

    return godag, annotations, associations, term_count, namespace


__all__ = [
    "initialize_go",
    "name2terms",
    "name2terms_counts",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GO data initialization for different namespaces")
    parser.add_argument(
        "--namespace",
        type=str,
        choices=["BP", "MF", "CC"],
        default="BP",
        help="GO namespace: BP, MF, or CC",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory for the GO data",
    )

    args = parser.parse_args()

    godag, annotations, associations, term_count, namespace = initialize_go(
        namespace_code=args.namespace,
        data_dir=args.data_dir,
    )

    print(f"\nGO {namespace} Statistics:")
    print(f"  - Number of annotations: {len(annotations)}")
    print(f"  - Number of annotated genes: {len(associations)}")