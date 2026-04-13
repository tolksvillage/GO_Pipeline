import os
import json
import argparse
import subprocess
import time
import platform
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import requests


def load_manifold_analysis(json_path: str) -> Dict:
    """Load a manifold_analysis JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_terms_with_robustness(data: Dict) -> List[Dict]:
    """
    Extract all GO terms with their robustness score (frequency_percentage).
    """
    terms = []

    for group in data.get("groups", []):
        for term in group.get("terms", []):
            term_info = {
                "go_id": term.get("go_id"),
                "name": term.get("name"),
                "definition": term.get("definition", "No definition available"),
                "robustness_score": term.get("frequency_percentage", 0.0),
                "group_id": group.get("group_id"),
                "group_size": group.get("size"),
            }
            terms.append(term_info)

    terms.sort(key=lambda x: x["robustness_score"], reverse=True)
    return terms


def format_terms_for_llm(terms: List[Dict]) -> str:
    """Format terms for LLM input."""
    formatted = "# Ontological Terms with Robustness Scores\n\n"

    for i, term in enumerate(terms, 1):
        formatted += f"{i}. {term['name']} ({term['go_id']})\n"
        formatted += f"   - Robustness Score: {term['robustness_score']:.1f}%\n"
        formatted += f"   - Definition: {term['definition']}\n"
        formatted += f"   - Group ID: {term['group_id']} (Group Size: {term['group_size']})\n\n"

    return formatted


def create_llm_prompt(terms_text: str, signature_name: str, ontology: str) -> str:
    """Create the full prompt for the LLM."""
    prompt = f"""You are an expert in the interpretation of gene signatures and ontological enrichment results.

I will provide you with a list of ontological terms, each accompanied by a robustness score (0–100%), indicating its relative importance within the signature.

{terms_text}

Your tasks are:
1. Analyze the terms and describe how they relate to each other. Determine the key underlying biological theme or mechanism suggested by these terms.

2. Write a concise and accurate summary that captures the core biological insight from this set of terms.

3. Propose two titles for the signature:
   * A detailed, specific title that reflects the identified biological mechanisms or processes.
   * A broader, high-level title that places these mechanisms within a wider biological or physiological context.

Ensure clarity, precision, and scientific depth in your response.

CRITICAL FORMATTING RULES:
- For titles: Provide only the title text itself, without quotation marks, asterisks, bold formatting, or explanatory text
- Do not include phrases like "This title captures..." or "This reflects..." after the titles
- Each title should be a single standalone phrase or sentence
- The summary should be a clear paragraph of text

Please structure your response exactly as follows:
## Relationship Analysis
[Your analysis of how terms relate]

## Summary
[Your concise summary paragraph]

## Proposed Titles
### Detailed Title
[Only the specific title text, no formatting or explanation]

### Broader Title
[Only the broader title text, no formatting or explanation]
"""
    return prompt


def query_ollama(
    prompt: str,
    model: str = "llama3.1:8b",
    ollama_url: str = "http://localhost:11434",
) -> str:
    """
    Send a request to Ollama and return the response.
    """
    url = f"{ollama_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 2000,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    except requests.exceptions.RequestException as e:
        print(f"Error during Ollama request: {e}")
        return ""


def save_analysis(
    signature_name: str,
    ontology: str,
    analysis_text: str,
    results_dict: Dict,
):
    """
    Store the parsed analysis in the results dictionary.
    """
    parsed = parse_llm_response(analysis_text, is_synthesis=False)

    if signature_name not in results_dict:
        results_dict[signature_name] = {}

    results_dict[signature_name][ontology] = parsed



def save_synthesis(
    signature_name: str,
    synthesis_text: str,
    results_dict: Dict,
):
    """
    Store the final synthesis in the results dictionary.
    """
    parsed = parse_llm_response(synthesis_text, is_synthesis=True)

    if signature_name not in results_dict:
        results_dict[signature_name] = {}

    results_dict[signature_name]["final"] = parsed



def save_signature_json(
    signature_name: str,
    signature_data: Dict,
    signature_path: Path,
    output_subdir: str,
):
    """
    Save the results of one signature as a JSON file inside the signature directory.
    """
    output_path = signature_path / output_subdir
    output_path.mkdir(parents=True, exist_ok=True)

    json_filepath = output_path / "llm_analysis.json"

    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(signature_data, f, indent=2, ensure_ascii=False)



def create_synthesis_prompt(ontology_summaries: Dict[str, str]) -> str:
    """
    Create the prompt for synthesizing all ontology analyses.
    """
    summaries_text = ""
    ontology_names = {
        "BP": "Biological Process",
        "MF": "Molecular Function",
        "CC": "Cellular Component",
    }

    for i, (ontology, summary) in enumerate(sorted(ontology_summaries.items()), 1):
        ontology_full_name = ontology_names.get(ontology, ontology)
        summaries_text += f"## Summary {i}: {ontology_full_name} Analysis\n\n"
        summaries_text += f"{summary}\n\n"
        summaries_text += "---\n\n"

    prompt = f"""You are an expert in interpreting gene signatures. I will provide you with {len(ontology_summaries)} independent summaries, each derived from a separate ontological analysis of the same gene signature.

{summaries_text}

Your task is to:
1. Synthesize the summaries into a unified, comprehensive interpretation of the gene signature. Identify the common thread or overarching biological mechanism that connects the insights from the individual analyses.

2. Write a precise and cohesive final summary that clearly explains the core biological implication of the gene signature as a whole.

3. Propose two final titles for the gene signature:
   * A detailed title that reflects the specific biological themes or mechanisms uncovered across the analyses.
   * A broader, high-level title that places the signature within a wider biological, physiological, or disease-related context.

Ensure your response is clear, well-structured, scientifically accurate, and deeply insightful.

CRITICAL FORMATTING RULES:
- For titles: Provide only the title text itself, without quotation marks, asterisks, bold formatting, or explanatory text
- Do not include phrases like "This title captures..." or "This reflects..." after the titles
- Each title should be a single standalone phrase or sentence
- The summary should be a clear paragraph of text

Please structure your response exactly as follows:
## Synthesis of Ontological Analyses
[Your integrated interpretation connecting all analyses]

## Final Summary
[Your comprehensive final summary of the gene signature]

## Proposed Final Titles
### Detailed Title
[Only the specific title text, no formatting or explanation]

### Broader Title
[Only the broader title text, no formatting or explanation]
"""
    return prompt


def extract_summary_from_analysis(analysis_text: str) -> str:
    """
    Extract the summary section from an LLM analysis.
    Looks for the '## Summary' section.
    """
    lines = analysis_text.split("\n")
    summary_lines = []
    in_summary = False

    for line in lines:
        if line.strip().startswith("## Summary"):
            in_summary = True
            continue
        elif line.strip().startswith("##") and in_summary:
            break
        elif in_summary:
            summary_lines.append(line)

    summary = "\n".join(summary_lines).strip()

    if not summary:
        summary = analysis_text

    return summary


def parse_llm_response(response_text: str, is_synthesis: bool = False) -> Dict:
    """
    Parse the LLM response and extract structured information.
    """
    result = {
        "summary": "",
        "detailed_title": "",
        "general_title": "",
    }

    lines = response_text.split("\n")
    current_section = None
    current_content = []

    if is_synthesis:
        summary_markers = ["## Final Summary", "## Summary"]
        detailed_markers = ["### Detailed Title"]
        general_markers = ["### Broader Title", "### High-level Title"]
    else:
        summary_markers = ["## Summary"]
        detailed_markers = ["### Detailed Title"]
        general_markers = ["### Broader Title", "### High-level Title"]

    for line in lines:
        line_stripped = line.strip()

        if any(marker in line_stripped for marker in summary_markers):
            if current_section and current_content:
                result[current_section] = "\n".join(current_content).strip()
            current_section = "summary"
            current_content = []

        elif any(marker in line_stripped for marker in detailed_markers):
            if current_section and current_content:
                result[current_section] = "\n".join(current_content).strip()
            current_section = "detailed_title"
            current_content = []

        elif any(marker in line_stripped for marker in general_markers):
            if current_section and current_content:
                result[current_section] = "\n".join(current_content).strip()
            current_section = "general_title"
            current_content = []

        elif line_stripped.startswith("##") or line_stripped.startswith("###"):
            continue

        elif current_section and line_stripped:
            current_content.append(line)

    if current_section and current_content:
        result[current_section] = "\n".join(current_content).strip()

    result["detailed_title"] = clean_title(result["detailed_title"])
    result["general_title"] = clean_title(result["general_title"])

    return result


def clean_title(title: str) -> str:
    """
    Clean a title string by removing unwanted formatting and explanations.
    """
    if not title:
        return ""

    title = title.replace("**", "")
    title = title.replace("*", "")
    title = title.replace("__", "")
    title = title.replace("_", "")

    title = title.strip("\"'")

    lines = title.split("\n")
    title = lines[0].strip()

    explanation_phrases = [
        "This title",
        "This reflects",
        "This captures",
        "Note:",
        "This emphasizes",
        "This broader",
        "This detailed",
        "The term",
        "While the",
        "This places",
    ]

    for phrase in explanation_phrases:
        if phrase in title:
            idx = title.find(phrase)
            title = title[:idx].strip()
            break

    title = title.rstrip(".")
    title = title.strip()

    if "\n" in title or len(title) > 200:
        import re

        sentences = re.split(r"[.!?]\s+", title)
        if sentences:
            title = sentences[0].strip()

    return title


def synthesize_signature_analysis(
    signature_name: str,
    results_dict: Dict,
    model: str,
    ollama_url: str,
) -> Tuple[bool, str]:
    """
    Create a final synthesis from all ontology analyses of one signature.
    """
    if signature_name not in results_dict:
        return False, f"No analyses found for {signature_name}"

    signature_data = results_dict[signature_name]

    ontology_summaries = {}
    for key, value in signature_data.items():
        if key != "final" and isinstance(value, dict) and "summary" in value:
            ontology_summaries[key] = value["summary"]

    if not ontology_summaries:
        return False, f"No ontology analyses found for {signature_name}"

    if len(ontology_summaries) < 2:
        return False, f"At least 2 ontology analyses required, found: {len(ontology_summaries)}"

    synthesis_prompt = create_synthesis_prompt(ontology_summaries)

    llm_response = query_ollama(synthesis_prompt, model=model, ollama_url=ollama_url)

    if not llm_response:
        return False, "No response received from Ollama"

    save_synthesis(signature_name, llm_response, results_dict)

    return True, ""


def process_signature(
    signature_path: Path,
    ontology: str,
    model: str,
    ollama_url: str,
    results_dict: Dict,
) -> Tuple[bool, str]:
    """
    Process a single signature-ontology combination.
    """
    manifold_file = (
        signature_path / "parameter_analysis" / ontology / f"manifold_analysis_{ontology}.json"
    )

    if not manifold_file.exists():
        return False, f"File not found: {manifold_file}"

    try:
        data = load_manifold_analysis(str(manifold_file))
        signature_name = data.get("metadata", {}).get("signature_name", signature_path.name)

        terms = extract_terms_with_robustness(data)

        if not terms:
            return False, f"No terms found in {manifold_file}"

        terms_text = format_terms_for_llm(terms)
        prompt = create_llm_prompt(terms_text, signature_name, ontology)

        llm_response = query_ollama(prompt, model=model, ollama_url=ollama_url)

        if not llm_response:
            return False, "No response received from Ollama"

        save_analysis(signature_name, ontology, llm_response, results_dict)

        return True, ""

    except Exception as e:
        return False, f"Processing error: {str(e)}"


def find_signatures(input_dir: str) -> List[Path]:
    """Find all signature directories in the input directory."""
    input_path = Path(input_dir)

    signatures = []
    for item in input_path.iterdir():
        if not item.is_dir():
            continue

        if item.name.startswith("diluted_"):
            continue

        param_analysis_dir = item / "parameter_analysis"
        if param_analysis_dir.exists() and param_analysis_dir.is_dir():
            signatures.append(item)

    return signatures


def find_ontologies(signature_path: Path) -> List[str]:
    """Find all available ontologies for a signature."""
    param_analysis_dir = signature_path / "parameter_analysis"

    ontologies = []
    for item in param_analysis_dir.iterdir():
        if item.is_dir():
            manifold_file = item / f"manifold_analysis_{item.name}.json"
            if manifold_file.exists():
                ontologies.append(item.name)

    return ontologies


def is_ollama_running(ollama_url: str) -> bool:
    """Check whether the Ollama server is reachable."""
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except Exception:
        return False


def start_ollama_server() -> bool:
    """Start Ollama in the background if it is not already running."""
    try:
        system = platform.system()

        if system == "Windows":
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        return True
    except Exception as e:
        print(f"Failed to start Ollama automatically: {e}")
        return False


def ensure_ollama_running(ollama_url: str, wait_seconds: int = 10) -> bool:
    """Ensure that the Ollama server is running, start it if necessary."""
    if is_ollama_running(ollama_url):
        return True


    if not start_ollama_server():
        return False

    for _ in range(wait_seconds):
        time.sleep(1)
        if is_ollama_running(ollama_url):
            return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Analyze gene signatures with an Ollama LLM"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing signature results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="llm_analysis",
        help="Name of the output subdirectory inside each signature directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
        help="Ollama model",
    )
    parser.add_argument(
        "--ollama_url",
        type=str,
        default="http://localhost:11434",
        help="URL of the Ollama server",
    )
    parser.add_argument(
        "--ontology",
        type=str,
        default=None,
        help="Specific ontology to process (for example BP, MF, CC)",
    )
    parser.add_argument(
        "--signature",
        type=str,
        default=None,
        help="Specific signature to process",
    )
    parser.add_argument(
        "--synthesize_only",
        action="store_true",
        help="Run synthesis only and skip ontology analyses",
    )
    parser.add_argument(
        "--skip_synthesis",
        action="store_true",
        help="Skip synthesis and run ontology analyses only",
    )

    args = parser.parse_args()

    if not ensure_ollama_running(args.ollama_url):
        print(f"Error: Ollama server is not reachable at {args.ollama_url}")
        print("Please make sure Ollama is installed and available in your PATH.")
        return

    signatures = find_signatures(args.input_dir)

    if not signatures:
        print(f"No signatures found in {args.input_dir}")
        return

    if args.signature:
        signatures = [s for s in signatures if args.signature in s.name]
        if not signatures:
            print(f"Signature '{args.signature}' not found")
            return

    all_results = {}

    total_processed = 0


    for signature_path in tqdm(signatures, desc="Signatures", unit="sig"):

        if args.synthesize_only:
            synthesize_signature_analysis(
                signature_path.name,
                all_results,
                args.model,
                args.ollama_url,
            )

            continue

        ontologies = find_ontologies(signature_path)

        if args.ontology:
            ontologies = [o for o in ontologies if o == args.ontology]

        if not ontologies:
            print("  No ontologies found")
            continue

        for ontology in ontologies:

            process_signature(
                signature_path,
                ontology,
                args.model,
                args.ollama_url,
                all_results,
            )

            total_processed += 1


        if not args.skip_synthesis and len(ontologies) >= 2:
            synthesize_signature_analysis(
                signature_path.name,
                all_results,
                args.model,
                args.ollama_url,
            )


        if signature_path.name in all_results:
            save_signature_json(
                signature_path.name,
                all_results[signature_path.name],
                signature_path,
                args.output_dir,
            )



if __name__ == "__main__":
    main()