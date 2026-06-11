# GO Pipeline

A Python pipeline for processing and analyzing gene signatures using the Gene Ontology (GO).

---

## Installation

### 0. Install LSF Git for large data

```bash
git lfs install
```

### 1. Clone the repository

```bash
git clone https://github.com/tolksvillage/GO_Pipeline.git
cd GO_Pipeline
```

### 2. Create and activate a Conda environment

```bash
conda create -n geneontology python=3.12
conda activate geneontology
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama
https://ollama.com

### 5. Load model
ollama pull llama3.1:8b

model can be changed in query_ollama function (llm_request.py) 

---

## Usage

```bash
python main.py --signatures_path <PATH> --output_path <PATH> --with_dilution --dilution_mode mode --with_llm_request true --with_paths true
```

with `mode` is `fixed, cumulative` or `both`.


# GO Term Analysis Pipeline – Usage Guide

## Quick Start

```bash
python main.py --output_path=<PATH> --signatures_path=<PATH>
```

These two arguments are the only ones required for a basic analysis run.

---

## Input Format

### `--signatures_path`

Path to a directory containing one or more gene signature files in plain-text format.

**Example path:**
```
C:/Username/Desktop/signatures
```

**File format** (one gene symbol per line):
```
GENESYMBOL_1
GENESYMBOL_2
...
GENESYMBOL_N
```

Each `.txt` file in the directory is treated as one independent gene signature. The filename (without extension) becomes the signature name used throughout all outputs.

---

### `--output_path`

Path to the root directory where all results will be written.

```
C:/Username/Desktop/results
```

> If the directory does not exist, it will be created automatically.

---

## Minimal Example

```bash
python main.py \
  --signatures_path=C:/Username/Desktop/signatures \
  --output_path=C:/Username/Desktop/results
```

---

## Optional Analyses

### Dilution Analysis

The dilution analysis tests how robustly each GO term survives progressive contamination of the input signature with random genes. To enable it, add the `--with_dilution` flag together with a dilution mode.

```bash
python main.py \
  --signatures_path=C:/Username/Desktop/signatures \
  --output_path=C:/Username/Desktop/results \
  --with_dilution \
  --dilution_mode cumulative
```

#### `--dilution_mode`

| Value | Description |
|---|---|
| `cumulative` | Each dilution step adds another batch of random genes on top of all previous ones. The signature gets progressively noisier with every step |
| `fixed` | Each dilution step replaces the random portion with a new, non-overlapping batch of the same fixed size. The noise level stays constant across steps |
| `both` | Runs both modes sequentially |

---

### Work in Progress

The following flags are implemented but not yet ready for production use. **Do not execute them in the current version.**

| Flag | Description |
|---|---|
| `--with_llm_request true` | Fetches gene and GO term descriptions from NCBI and GeneCards so that a local LLM can generate biological summaries |
| `--with_paths true` | For each term enriched by the robustness analysis, computes the optimal paths through the GO ontology from the root term down to the enriched term |

---

## Output Directory Structure

After a successful run, the following directory structure is created under `--output_path`.

> **Note:** Folders prefixed with `diluted_` are auxiliary working directories created during the dilution pipeline. They were originally used for debugging purposes and can be ignored.

---

### Core Analysis (without Dilution)

```text
{output_path}/
└── {signature_name}/
     └── parameter_analysis/
          ├── BP/
          │    ├── manifold_analysis_BP.json
          │    └── manifold_visualization/
          │         └── manifold_BP_simple.png
          ├── MF/
          │    ├── manifold_analysis_MF.json
          │    └── manifold_visualization/
          │         └── manifold_MF_simple.png
          └── CC/
               ├── manifold_analysis_CC.json
               └── manifold_visualization/
                    └── manifold_CC_simple.png
```

A separate subfolder is created for each of the three GO ontologies (`BP`, `MF`, `CC`).

| Path                                                    | Description                                                                                                                                                                             |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `manifold_analysis_{ontology}.json`                     | Complete analysis results: all enriched GO terms, their robustness scores, α/β preferences, gene associations, and inter-group connections. Full data structure documented separately   |
| `manifold_visualization/manifold_{ontology}_simple.png` | Visual representation of the solution set. Enriched terms are distributed across three layers according to their α/β preference (Core Function Layer, Integration Layer, Context Layer) |

#### Example for EMT-Hallmark signature

Manifold `JSON` for biological processes:

![[manifold_analysis_BP.json]]

Manifold Visualization: 

![[manifold_BP_simple.png]]

For explanation look at `manifold_analysis_json_data_structure.md`!

---

### Dilution Analysis

```text
{output_path}/
└── {signature_name}/
     └── Dilutions/
          ├── BP/
          │    ├── cumulative/
          │    │    ├── {signature_name}_ranking.json
          │    │    └── robustness_heatmap_{signature_name}.png
          │    └── fixed/
          │         ├── {signature_name}_ranking.json
          │         └── robustness_heatmap_{signature_name}.png
          ├── MF/
          │    └── ...
          └── CC/
               └── ...
```

A separate subfolder is created per ontology and per dilution mode.

| File                                      | Description                                                                                                                                                                                                                                                                         |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `{signature_name}_ranking.json`           | All GO terms from the original signature ranked by their robustness score. Tracks how the `robustness_score` of each term evolves across dilution steps and computes a composite dilution resistance score. Full data structure documented separately                               |
| `robustness_heatmap_{signature_name}.png` | Heatmap visualising `robustness_score` per GO term across all dilution steps. Rows are sorted by `score_components.dilution_resistance_score`. If automatic cutoff detection was enabled, the cutoff boundary is shown as a dashed red line separating robust from non-robust terms |

> **Note on the `fixed` mode output:** The `fixed` mode directory may contain additional intermediate files beyond the two listed above. These are artefacts of the fixed-mode pipeline and do not need to be considered for downstream analysis.



#### Example for EMT-Hallmark signature

Ranking `JSON` for biological processes:

![[hallmark_EMT_ranking.json]]

Heatmap visualization `fixed`:

![[robustness_heatmap_hallmark_EMT 1.png]]


Heatmap visualization `cumulative`:

![[robustness_heatmap_hallmark_EMT 2.png]]

For explanation look at `dilution_analysis_ranking_json_data_structure.md`!

### Intermediate Files

The pipeline also creates several intermediate files used by later steps, including gene-to-GO mappings, reduced representative terms, hierarchy maps, parameter sweep results, and robust validation files. These files are mainly used internally by downstream scripts.

e.g.

{signature}/BP/my_terms_BP.txt {signature}/BP/mapping_genes_to_bp/map_genes_to_bp.txt {signature}/representatives_analysis/reduced_terms_bp.json {signature}/representatives_analysis/hierarchy_map_bp.json {signature}/representatives_analysis/gene_assignments_bp.json {signature}/representatives_analysis/rep_to_genes_bp.json {signature}/BP/keyword_analysis/BP_parameter_analysis_sum.json {signature}/robust_terms_validation.json


# Masterthesis

For a complete mathematical description of the algorithms, please refer to the master's thesis.!

[[Masterthesis.pdf]]

Note: The original work was written in German and subsequently translated into English using AI. Any formatting inconsistencies, translation inaccuracies, or linguistic irregularities that may appear are the result of the automated translation process rather than the original content.

# Recommandation

Recommendation: For the introduction, it is important to become familiar with the basic concepts. For this purpose, please refer to [[Masterthesis.pdf]] p. 4-21.