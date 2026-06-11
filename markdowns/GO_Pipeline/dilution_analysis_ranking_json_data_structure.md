
#### Complete mathematical description in [[Masterthesis.pdf]] p. 46-48

## Overview

The `ranking.json` is produced by the **GO Terms Extractor** (`dilute_analysis.py`) as the primary output of the dilution robustness analysis. It is written to:

```text
{signature}/Dilutions/{ontology}/{mode}/{signature}_ranking.json
```

The file answers the question: *How consistently does a GO term remain in the Top-10 of the parameter analysis when the input gene signature is progressively diluted with random genes?*

---

## How the File is Generated – Pipeline Overview

```text
1. dilute_analysis.py
   └── Creates diluted signature .txt files
        (original + N × random genes, cumulative or fixed)

2. Runs GO term scoring across all (α, β) configurations
        for each diluted signature
        → manifold_analysis_{ontology}.json per diluted variant

3. Loads all manifold JSONs for one base signature
   └── Tracks robustness_score of each GO term across dilution steps
   └── Computes dilution_resistance_score per term
   └── Detects automatic cutoff (max-gap method)
   └── Writes ranking.json + heatmap PNG
```

---

## Hierarchy Overview

```text
root
├── metadata
├── score_calculation_formula
│    ├── formula
│    └── components
│         ├── mean_retention
│         ├── stability
│         └── avg_deviation
└── ranking[]
     ├── rank
     ├── go_id
     ├── go_name
     ├── ic
     ├── genes_direct
     ├── genes_inherited
     ├── genes_total
     ├── is_robust
     ├── robustness_score
     │    ├── Original
     │    ├── Step_01
     │    ├── Step_02
     │    └── ...
     └── score_components
          ├── initial_robustness
          ├── final_robustness
          ├── mean_robustness
          ├── mean_retention
          ├── avg_deviation
          ├── stability
          └── dilution_resistance_score
```

---

## Top-Level Structure

```json
{
  "metadata": {...},
  "score_calculation_formula": {...},
  "ranking": [...]
}
```

| Field                       | Type   | Description                                                                                                                                                                                                                        |
| --------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `metadata`                  | object | Provenance and configuration of this analysis run                                                                                                                                                                                  |
| `score_calculation_formula` | object | Human-readable documentation of how `dilution resistance score` is computed                                                                                                                                                        |
| `ranking`                   | array  | All GO terms from the original signature ranked by `score_components.dilution_resistance_score`. Tracks how the `robustness_score` of each term evolves across dilution steps and computes a composite dilution resistance score.  |

---

## `metadata`

```json
{
  "signature_name": "hallmark_EMT",
  "dilution_mode": "cumulative",
  "analysis_type": "all_terms",
  "threshold": 0.0,
  "n_terms": 21,
  "n_steps": 11,
  "auto_cutoff_enabled": true,
  "auto_cutoff_rank": 4
}
```


| Field | Type | Description |  
|---|---|---|  
| `signature_name` | string | Name of the base gene signature, taken from the signature directory name |  
| `dilution_mode` | string | `"cumulative"`: each step adds another batch of random genes on top of all previous ones; `"fixed"`: each step uses a new, non-overlapping random batch of equal size |  
| `analysis_type` | string | `"all_terms"`: all unique GO terms found in the original signature's manifold were tracked; `"representatives"`: only the group-representative terms were tracked |  
| `threshold` | float | Minimum `robustness_score["Original"]` required for a term to be included. `0.0` means all terms are included regardless of initial robustness |  
| `n_terms` | integer | Total number of GO terms included in this ranking |  
| `n_steps` | integer | Total number of data points per term: 1 original + N dilution steps |  
| `auto_cutoff_enabled` | boolean | Whether automatic cutoff detection was enabled |  
| `auto_cutoff_rank` | integer or null | Rank at which the largest absolute gap in `score_components.dilution_resistance_score` was detected. Terms with `rank <= auto_cutoff_rank` are flagged as `is_robust: true`. `null` if `auto_cutoff_enabled` is `false` |
---

## `score_calculation_formula`

Documents the dilution resistance scoring logic inline in the JSON for reproducibility.

```json
{
  "formula": "dilution_resistance_score = initial_robustness * mean_retention * stability",
  "components": {
    "mean_retention": "mean(all_robustness) / initial_robustness",
    "stability": "max(0, 1 - (avg_deviation / initial_robustness))",
    "avg_deviation": "mean(|robustness_i - initial_robustness|) for i > 0"
  }
}
```

The full formula is:

$$\text{dilution\_resistance\_score} = r_0 \times \underbrace{\frac{\bar{r}}{r_0}}_{\text{mean\_retention}} \times \underbrace{\max\!\left(0,\; 1 - \frac{\overline{|r_i - r_0|}}{r_0}\right)}_{\text{stability}}$$

where $r_0$ is the robustness score in the original undiluted signature and $r_i$ are the robustness scores at each dilution step $i > 0$.

- **`mean_retention`** measures how much of the original robustness is preserved on average across all steps, including the original. A value of `1.0` means the term never lost robustness on average. 
- **`avg_deviation`** measures how volatile the robustness score is across dilution steps relative to the starting point (steps $i > 0$ only).
- **`stability`** penalises high volatility. A term that fluctuates heavily even if it survives on average will receive a lower stability score. Clamped to `0` from below.
- The final **`dilution_resistance_score`** is in the same unit as `robustness_score` (0–100), so it is directly interpretable as a weighted effective robustness value.

---

## `ranking[]`

Each entry represents one GO term tracked across all dilution steps. Entries are **sorted by `score_components.dilution_resistance_score` descending**, so the most dilution-resistant term is `rank 1`.

```json
{
  "rank": 1,
  "go_id": "GO:0030198",
  "go_name": "extracellular matrix organization",
  "ic": 0.4598,
  "genes_direct": 16,
  "genes_inherited": 37,
  "genes_total": 53,
  "is_robust": true,
  "robustness_score": {...},
  "score_components": {...}
}
```

| Field              | Type            | Description                                                                                                                                        |
| ------------------ | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rank`             | integer         | 1-based rank by `score_components.dilution_resistance_score` descending                                                                            |
| `go_id`            | string          | GO accession number                                                                                                                                |
| `go_name`          | string          | Human-readable GO term name                                                                                                                        |
| `ic`               | float           | Information Content of this term, taken from the first parameter configuration in which the term appeared in the original (undiluted) manifold     |
| `genes_direct`     | integer         | Number of genes directly annotated to this term in the original signature                                                                          |
| `genes_inherited`  | integer         | Number of genes inherited from child terms in the original signature                                                                               |
| `genes_total`      | integer         | `genes_direct + genes_inherited`                                                                                                                   |
| `is_robust`        | boolean \| null | `true` if `rank ≤ auto_cutoff_rank`; `false` if `rank > auto_cutoff_rank`. If automatic cutoff detection was disabled, all terms are marked `true` |
| `robustness_score` | object          | Robustness score of this term at each dilution step                                                                                                |
| `score_components` | object          | Intermediate values and the final `dilution_resistance_score`                                                                                      |

---

## `robustness_score`

Tracks the `robustness_score` of the term (i.e. in how many of the 120 `(α, β)` parameter configurations it appeared in the Top-10) at each dilution step.

```json
{
  "Original": 100.0,
  "Step_01": 100.0,
  "Step_02": 91.67,
  "Step_03": 100.0,
  "Step_04": 100.0,
  "Step_05": 91.67,
  "Step_06": 88.33,
  "Step_07": 88.33,
  "Step_08": 81.67,
  "Step_09": 83.33,
  "Step_10": 85.83
}
```

| Key                  | Description                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Original`           | Robustness score in the unmodified signature, from `manifold_analysis_{ontology}.json`                                                                                                                                                                                                                                                                                                           |
| `Step_01` … `Step_N` | Robustness score after adding the N-th batch of random genes. In **cumulative** mode each step adds another batch of size equal to the original signature size on top of all previous additions. In **fixed** mode each step replaces the random portion with a new, non-overlapping batch of the same size. If the term was not present in the manifold at a given step it is recorded as `0.0` |

The number of dilution steps is controlled by `--steps`. The number of random genes added per step is set internally to the size of the original signature.

---

## `score_components`

All intermediate values used to compute `dilution_resistance_score`, stored for full transparency.

```json
{
  "initial_robustness": 100.0,
  "final_robustness": 85.83,
  "mean_robustness": 91.14,
  "mean_retention": 0.9114,
  "avg_deviation": 9.75,
  "stability": 0.9025,
  "dilution_resistance_score": 82.25
}
```

| Field                       | Type  | Description                                                                                                                                                                                         |
| --------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `initial_robustness`        | float | `robustness_score["Original"]` – the anchor value for all subsequent calculations                                                                                                                   |
| `final_robustness`          | float | `robustness_score["Step_N"]` – robustness score at the last dilution step; purely informational, not used directly in the score                                                                     |
| `mean_robustness`           | float | Arithmetic mean of **all** steps including `Original`                                                                                                                                               |
| `mean_retention`            | float | `mean_robustness / initial_robustness`. Values `> 1` are possible if the term gains robustness during dilution, for example when random noise happens to add co-annotated genes.                    |
| `avg_deviation`             | float | Mean absolute deviation from `initial_robustness` computed over steps $i > 0$ only: $\overline{\|r_i - r_0\|}$                                                                                      |
| `stability`                 | float | `max(0, 1 − avg_deviation / initial_robustness)`. Clamped to `0` if `avg_deviation ≥ initial_robustness`. Will also be `0` if `initial_robustness = 0`.                                             |
| `dilution_resistance_score` | float | Final composite score: `initial_robustness × mean_retention × stability`. Range is nominally 0–100 but can slightly exceed 100 if `mean_retention > 1`. A small score indicates extreme volatility. |

---

## Automatic Cutoff Detection

When `auto_cutoff_enabled: true`, the extractor applies the **maximum absolute gap** method after sorting all terms by `score_components.dilution_resistance_score`:

1. Compute all consecutive score differences: $\Delta_i = \text{score}_i - \text{score}_{i+1}$
2. Identify the index $i^*$ with the largest $\Delta_{i^*}$
3. Set `auto_cutoff_rank = i* + 1`

Terms at `rank ≤ auto_cutoff_rank` receive `is_robust: true`; all others receive `is_robust: false`. This threshold is also rendered as a dashed red line in the heatmap output.