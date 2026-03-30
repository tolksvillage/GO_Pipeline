import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def filter_manifold_by_alpha_beta_preference(data, preference_type, threshold=0.15):
    """
    Filter manifold to keep only terms matching the specified alpha-beta preference.

    preference_type: 'alpha_greater', 'beta_greater', or 'equal'
    threshold: normalized difference threshold (default: 0.15 = 15%)

    Classification based on difference_normalized:
    - alpha_greater: difference_normalized > threshold
    - beta_greater: difference_normalized < -threshold
    - equal: abs(difference_normalized) <= threshold
    """
    filtered_data = {
        'metadata': data.get('metadata', {}),
        'summary_statistics': {},
        'groups': []
    }

    total_terms = 0

    for group in data['groups']:
        filtered_terms = []

        for term in group['terms']:
            alpha_beta_pref = term.get('alpha_beta_preference', {})
            difference_normalized = alpha_beta_pref.get('difference_normalized', 0.0)

            keep_term = False
            if preference_type == 'alpha_greater' and difference_normalized > threshold:
                keep_term = True
            elif preference_type == 'beta_greater' and difference_normalized < -threshold:
                keep_term = True
            elif preference_type == 'equal' and abs(difference_normalized) <= threshold:
                keep_term = True

            if keep_term:
                filtered_terms.append(term)

        if filtered_terms:
            filtered_group = {
                'group_id': group['group_id'],
                'original_group_index': group['original_group_index'],
                'size': len(filtered_terms),
                'representative': group['representative'],
                'terms': filtered_terms,
                'connections_to_other_groups': group.get('connections_to_other_groups', [])
            }
            filtered_data['groups'].append(filtered_group)
            total_terms += len(filtered_terms)

    filtered_data['summary_statistics'] = {
        'number_of_groups': len(filtered_data['groups']),
        'total_terms': total_terms,
        'filter_type': preference_type,
        'threshold': threshold
    }

    return filtered_data


def find_manifold_files(base_path):
    """Find all manifold analysis JSON files in the directory structure"""
    manifold_files = []

    search_path = Path(base_path)

    for signature_dir in search_path.iterdir():
        if not signature_dir.is_dir():
            continue

        param_analysis_dir = signature_dir / "parameter_analysis"
        if not param_analysis_dir.exists():
            continue

        for ontology in ['BP', 'MF', 'CC']:
            ontology_dir = param_analysis_dir / ontology

            if not ontology_dir.exists():
                continue

            json_file = ontology_dir / f"manifold_analysis_{ontology}.json"

            if json_file.exists():
                manifold_files.append({
                    'json_file': str(json_file),
                    'ontology': ontology,
                    'param_analysis_dir': str(param_analysis_dir),
                    'ontology_dir': str(ontology_dir),
                    'signature': signature_dir.name
                })

    return manifold_files


def process_single_manifold(manifold_info, threshold=0.15):
    """Process a single manifold file and add filtered versions to it"""
    json_file = manifold_info['json_file']

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'groups' not in data:
            return False

        alpha_greater_data = filter_manifold_by_alpha_beta_preference(data, 'alpha_greater', threshold)
        beta_greater_data = filter_manifold_by_alpha_beta_preference(data, 'beta_greater', threshold)
        equal_data = filter_manifold_by_alpha_beta_preference(data, 'equal', threshold)

        data['filtered_by_preference'] = {
            'threshold': threshold,
            'alpha_greater_than_beta': {
                'summary_statistics': alpha_greater_data['summary_statistics'],
                'groups': alpha_greater_data['groups']
            },
            'beta_greater_than_alpha': {
                'summary_statistics': beta_greater_data['summary_statistics'],
                'groups': beta_greater_data['groups']
            },
            'alpha_equals_beta': {
                'summary_statistics': equal_data['summary_statistics'],
                'groups': equal_data['groups']
            }
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return True

    except Exception:
        return False


def main():
    """Main function to extend manifold files with filtered versions"""

    parser = argparse.ArgumentParser(
        description="Extend manifold analysis files by adding alpha-beta preference filters",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--input_path', type=str, default='results',
                        help='Path to results directory')

    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.20,
        help="Normalized threshold for classifying alpha-beta preference (default: 0.15 = 15%%)"
    )

    args = parser.parse_args()

    input_path = args.input_path.strip()

    if not os.path.exists(input_path):
        print(f"Error: Base path '{input_path}' does not exist!")
        return 1

    manifold_files = find_manifold_files(input_path)

    successful_count = 0
    failed_count = 0

    progress_bar = tqdm(manifold_files, desc="Manifold files", unit="file", dynamic_ncols=True)

    for info in progress_bar:
        if process_single_manifold(info, threshold=args.threshold):
            successful_count += 1
        else:
            failed_count += 1

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    main()