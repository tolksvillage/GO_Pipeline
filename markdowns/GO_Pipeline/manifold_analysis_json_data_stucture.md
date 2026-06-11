#### Complete mathematical description in [[Masterthesis.pdf]] p. 37-39

## Hierarchy Overview from JSON

```text
root
├── metadata
├── summary_statistics
├── groups[]
│    ├── group_id
│    ├── original_group_index
│    ├── size
│    ├── representative
│    │    ├── go_id
│    │    ├── name
│    │    ├── robustness_score
│    │    ├── definition
│    │    └── gene_symbols
│    │         ├── GENE_A
│    │         ├── GENE_B
│    │         └── ...
│    │
│    ├── terms[]
│    │    ├── go_id
│    │    ├── name
│    │    ├── robustness_score
│    │    ├── alpha_beta_preference
│    │    │    ├── alpha_dominant
│    │    │    ├── beta_dominant
│    │    │    ├── equal
│    │    │    ├── difference_absolute
│    │    │    └── difference_normalized
│    │    ├── ic
│    │    ├── genes_direct
│    │    ├── genes_inherited
│    │    ├── genes_total
│    │    ├── path_ids
│    │    ├── definition
│    │    └── gene_symbols
│    │
│    └── connections_to_other_groups[]
│         ├── source_group_id
│         ├── target_group_id
│         ├── relationship_type
│         ├── source_term
│         └── target_term
│
└── filtered_by_preference
     ├── threshold
     ├── alpha_greater_than_beta
     │    ├── summary_statistics
     │    └── groups[]
     ├── beta_greater_than_alpha
     │    ├── summary_statistics
     │    └── groups[]
     └── alpha_equals_beta
          ├── summary_statistics
          └── groups[]
```

---

# Top-Level Structure

```json
{
  "metadata": {...},
  "summary_statistics": {...},
  "groups": [...],
  "filtered_by_preference": {...}
}
```

| Field | Type | Description |
|---|---|---|
| `metadata` | object | Analysis metadata |
| `summary_statistics` | object | Summary metrics of the full dataset |
| `groups` | array | All functional GO groups, unfiltered |
| `filtered_by_preference` | object | Filtered subsets partitioned by α/β preference; added by a separate post-processing step |

---

# `metadata`

```json
{
  "signature_name": "hallmark_EMT",
  "ontology": "BP",
  "analysis_type": "manifold_analysis"
}
```

| Field | Type | Description |
|---|---|---|
| `signature_name` | string | Name of the input gene signature (taken from the signature directory name) |
| `ontology` | string | GO ontology namespace (`BP`, `MF`, or `CC`) |
| `analysis_type` | string | Fixed identifier, always `"manifold_analysis"` |

---

# `summary_statistics`

```json
{
  "unique_go_terms_top10": 21,
  "parameter_configurations": 120,
  "diversity_index": 0.0175,
  "number_of_groups": 16
}
```

| Field | Type | Description |
|---|---|---|
| `unique_go_terms_top10` | integer | Total number of distinct GO terms that appeared in the Top-10 across **all** parameter configurations |
| `parameter_configurations` | integer | Total number of `(α, β)` parameter combinations that were evaluated |
| `diversity_index` | float | Ratio of unique GO terms to the theoretical maximum (`unique_terms / (n_configs × 10)`). A value near 0 means most configurations agree on the same terms; a value near 1 means every slot is occupied by a different term |
| `number_of_groups` | integer | Number of GO groups identified by hierarchical clustering via `is_a` parent–child relationships from the OBO ontology |

---

# `groups[]`

Each entry represents one functional GO cluster. Groups are **sorted in descending order by the `robustness_score` of their representative term**, so the most robust/prevalent group appears first.

```json
{
  "group_id": 1,
  "original_group_index": 3,
  "size": 4,
  "representative": {...},
  "terms": [...],
  "connections_to_other_groups": [...]
}
```

| Field                         | Type    | Description                                                                   |
| ----------------------------- | ------- | ----------------------------------------------------------------------------- |
| `group_id`                    | integer | Display index (1-based) after sorting by representative `robustness_score`    |
| `original_group_index`        | integer | Internal 0-based index assigned during graph traversal, before sorting        |
| `size`                        | integer | Number of GO terms in this group                                              |
| `representative`              | object  | The term with the highest `robustness_score` within the group                 |
| `terms`                       | array   | All GO terms belonging to this group, sorted by `robustness_score` descending |
| `connections_to_other_groups` | array   | Cross-group ontology relationships (e.g. `part_of`, `regulates`)              |

---

# `representative`

The term within the group that appeared most frequently across parameter configurations. It serves as the human-readable label for the group.

```json
{
  "go_id": "GO:0010811",
  "name": "positive regulation of cell-substrate adhesion",
  "robustness_score": 100.0,
  "definition": "...",
  "gene_symbols": {...}
}
```

| Field              | Type   | Description                                                                      |
| ------------------ | ------ | -------------------------------------------------------------------------------- |
| `go_id`            | string | GO accession number                                                              |
| `name`             | string | Human-readable GO term name                                                      |
| `robustness_score` | float  | Percentage of parameter configurations in which this term appeared in the Top-10 |
| `definition`       | string | Official GO term definition                                                      |
| `gene_symbols`     | object | Genes associated with this term (see `gene_symbols`)                             |
|                    |        |                                                                                  |

---

# `terms[]`

All GO terms within the group, each enriched with robustness, preference, and gene data.

```json
{
  "go_id": "GO:0010811",
  "name": "positive regulation of cell-substrate adhesion",
  "robustness_score": 87.5,
  "alpha_beta_preference": {...},
  "ic": 8.34,
  "genes_direct": 6,
  "genes_inherited": 12,
  "genes_total": 18,
  "path_ids": ["GO:0007155"],
  "definition": "...",
  "gene_symbols": {...}
}
```

| Field                   | Type             | Description                                                                                                                         |
| ----------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `go_id`                 | string           | GO accession number                                                                                                                 |
| `name`                  | string           | Human-readable GO term name                                                                                                         |
| `robustness_score`      | float            | Percentage of **all** parameter configurations in which this term appeared in the Top-10. Computed as `count / total_configs × 100` |
| `alpha_beta_preference` | object           | Counts and normalized difference describing under which α/β regime this term is preferred (see below)                               |
| `ic`                    | float            | Information Content of the term, taken from the first parameter configuration in which the term appeared                            |
| `genes_direct`          | integer          | Number of genes directly annotated to this term                                                                                     |
| `genes_inherited`       | integer          | Number of genes inherited from child terms                                                                                          |
| `genes_total`           | integer          | `genes_direct + genes_inherited`                                                                                                    |
| `path_ids`              | array of strings | All GO path IDs (primary + alternative) through which this term was reached, collected across all configurations                    |
| `definition`            | string           | Official GO term definition                                                                                                         |
| `gene_symbols`          | object           | Gene symbol → description mapping                                                                                                   |

---

# `alpha_beta_preference`

Describes in how many of the 120 parameter configurations this term was a Top-10 term **and** under which α/β weighting regime.

```json
{
  "alpha_dominant": 55,
  "beta_dominant": 55,
  "equal": 10,
  "difference_absolute": 0,
  "difference_normalized": 0.0
}
```

The scoring metric used in the underlying analysis is:

$$\text{score} = (\alpha \cdot \ln(\text{genes\_direct}) + \beta \cdot \ln(\text{genes\_inherited})) \times IC$$

- A high **α** therefore promotes terms with many **directly annotated** genes (specific, well-defined functions).
- A high **β** promotes terms with many **inherited** genes (broader, more general biological context).

| Field | Type | Description |
|---|---|---|
| `alpha_dominant` | integer | Number of Top-10 appearances in configurations where `α > β` |
| `beta_dominant` | integer | Number of Top-10 appearances in configurations where `β > α` |
| `equal` | integer | Number of Top-10 appearances in configurations where `α = β` |
| `difference_absolute` | integer | `alpha_dominant − beta_dominant`; positive = term prefers high-α regimes |
| `difference_normalized` | float | `difference_absolute / 55`. Normalized to `[−1, 1]` using 55 as the maximum possible one-sided count (there are exactly 55 configurations with `α > β` and 55 with `β > α`). A value of `+1` means the term appears exclusively under `α > β`; `−1` means exclusively under `β > α`; `0` means no preference |

---

# `gene_symbols`

Dictionary mapping gene symbol → description string.

```json
{
  "FN1": "fibronectin 1",
  "ITGA5": "integrin subunit alpha 5",
  "VEGFA": "vascular endothelial growth factor A"
}
```

---

# `connections_to_other_groups[]`

Cross-group ontology edges derived from the OBO file. A connection is recorded when a term in one group has an `is_a`, `part_of`, `regulates`, `positively_regulates`, or `negatively_regulates` relationship pointing to a term in a **different** group.

```json
{
  "source_group_id": 1,
  "target_group_id": 6,
  "relationship_type": "positively_regulates",
  "source_term": {
    "go_id": "GO:0010811",
    "name": "positive regulation of cell-substrate adhesion"
  },
  "target_term": {
    "go_id": "GO:0007155",
    "name": "cell adhesion"
  }
}
```

| Field | Type | Description |
|---|---|---|
| `source_group_id` | integer | `group_id` of the group containing the source term |
| `target_group_id` | integer | `group_id` of the group containing the target term |
| `relationship_type` | string | OBO relationship type (`is_a`, `part_of`, `regulates`, `positively_regulates`, `negatively_regulates`) |
| `source_term` | object | `go_id` and `name` of the term from which the relationship originates |
| `target_term` | object | `go_id` and `name` of the term to which the relationship points |

> **Note:** Only relationships between terms belonging to **different** groups are recorded here. Intra-group relationships (which define the group itself) are implicit in the grouping structure.

---

# `filtered_by_preference`

This section is **not** produced by the main manifold analysis. It is added to the same JSON file by a separate post-processing script that re-reads the file and appends these filtered views.

The partitioning is based on the `difference_normalized` value of each term and a configurable threshold `τ` (default `0.20`):

| Partition | Condition | Interpretation |
|---|---|---|
| `alpha_greater_than_beta` | `difference_normalized > τ` | Term appears predominantly under high-α (specificity-focused) configurations → **Core Function Layer** |
| `beta_greater_than_alpha` | `difference_normalized < −τ` | Term appears predominantly under high-β (context-focused) configurations → **Context Layer** |
| `alpha_equals_beta` | `|difference_normalized| ≤ τ` | Term appears without a strong preference → **Integration Layer** |

```json
{
  "filtered_by_preference": {
    "threshold": 0.20,
    "alpha_greater_than_beta": {
      "summary_statistics": {
        "number_of_groups": 4,
        "total_terms": 6,
        "filter_type": "alpha_greater",
        "threshold": 0.20
      },
      "groups": [...]
    },
    "beta_greater_than_alpha": {
      "summary_statistics": {...},
      "groups": [...]
    },
    "alpha_equals_beta": {
      "summary_statistics": {...},
      "groups": [...]
    }
  }
}
```

| Field                     | Type   | Description                                        |
| ------------------------- | ------ | -------------------------------------------------- |
| `threshold`               | float  | The `τ` value used for partitioning                |
| `alpha_greater_than_beta` | object | Groups/terms where `difference_normalized > τ`     |
| `beta_greater_than_alpha` | object | Groups/terms where `difference_normalized < −τ`    |
| `alpha_equals_beta`       | object | Groups/terms where `\|difference_normalized\| ≤ τ` |

Each partition contains its own `summary_statistics` and a `groups` array with the same structure as the top-level `groups[]`, but containing only the terms that passed the filter. The `size` field of each group reflects the number of **filtered** terms remaining, not the original group size.