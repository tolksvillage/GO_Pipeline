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

### 4. Ollama installieren
# separat von https://ollama.com

### 5. Modell laden
ollama pull llama3.1:8b

model can be changed in query_ollama function (llm_request.py) 

### 6. Optional: Ollama starten
ollama serve

---

## Usage

```bash
python main.py --signatures <PATH> --output_path <PATH> --with_dilution --dilution_mode
```
