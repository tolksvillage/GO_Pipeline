"""
Hierarchical Manifold Visualizer for GO Term Analysis
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import textwrap
import argparse
import sys
from tqdm import tqdm

# Configuration
DATA_SUBPATH = Path("parameter_analysis")
OUTPUT_SUBDIR = "manifold_visualization"


class HierarchicalManifoldVisualizer:
    def __init__(self):
        self._setup_colors()

    def _setup_colors(self):
        self.color_general = '#E8F4F8'
        self.color_bridge = '#FFF4E6'
        self.color_specific = '#F0F8F0'
        self.edge_color_general = '#2C5F7C'
        self.edge_color_bridge = '#D97706'
        self.edge_color_specific = '#15803D'


    def load_manifold_data(self, manifold_path: str) -> Dict:
        """Load manifold data from extended JSON file with filtered preferences"""
        with open(manifold_path, 'r') as f:
            full_data = json.load(f)

        if 'filtered_by_preference' not in full_data:
            raise ValueError(
                f"Manifold file {manifold_path} does not contain 'filtered_by_preference' section. "
            )

        filtered = full_data['filtered_by_preference']

        alpha_greater = {
            'metadata': full_data.get('metadata', {}),
            'summary_statistics': filtered['alpha_greater_than_beta']['summary_statistics'],
            'groups': filtered['alpha_greater_than_beta']['groups']
        }

        beta_greater = {
            'metadata': full_data.get('metadata', {}),
            'summary_statistics': filtered['beta_greater_than_alpha']['summary_statistics'],
            'groups': filtered['beta_greater_than_alpha']['groups']
        }

        equal = {
            'metadata': full_data.get('metadata', {}),
            'summary_statistics': filtered['alpha_equals_beta']['summary_statistics'],
            'groups': filtered['alpha_equals_beta']['groups']
        }

        return {
            'alpha_greater': alpha_greater,
            'beta_greater': beta_greater,
            'equal': equal
        }


    def _wrap_text(self, text: str, max_width: int = 25) -> str:
        """Helper to wrap text for node labels."""
        text = text.replace('\n', ' ')
        lines = textwrap.wrap(text, width=max_width, break_long_words=False)
        if len(lines) > 3:
            lines = lines[:2] + [lines[2][:max_width - 3] + '...']
        return '\n'.join(lines)

    def _extract_individual_terms_with_diff(self, groups: List[Dict]) -> List[Dict]:
        """Extract all individual terms from groups with their difference_normalized values"""
        all_terms = []
        for group in groups:
            if 'terms' in group and group['terms']:
                for term in group['terms']:
                    term_data = {
                        'name': term.get('name', 'Unknown'),
                        'go_id': term.get('go_id', ''),
                        'difference_normalized': term.get('alpha_beta_preference', {}).get('difference_normalized', 0.0),
                        'frequency_percentage': term.get('frequency_percentage', 0.0),
                        'group_id': group['group_id'],
                        'group_size': group['size']
                    }
                    all_terms.append(term_data)
            elif 'representative' in group:
                rep = group['representative']
                term_data = {
                    'name': rep.get('name', 'Unknown'),
                    'go_id': rep.get('go_id', ''),
                    'difference_normalized': rep.get('alpha_beta_preference', {}).get('difference_normalized', 0.0),
                    'frequency_percentage': rep.get('frequency_percentage', 0.0),
                    'group_id': group['group_id'],
                    'group_size': group['size']
                }
                all_terms.append(term_data)
        return all_terms

    def _calculate_y_position_from_diff(self, diff_normalized: float, zero_line_y: float, threshold: float = 0.2) -> float:
        diff_clamped = max(-1.0, min(1.0, diff_normalized))
        y_position = 0.5 - 0.5 * diff_clamped
        return y_position

    def _calculate_x_positions_ranked(self, terms: List[Dict], start_x: float = 0.08, layer_width: float = 0.84, box_width: float = 0.16) -> Dict[int, float]:

        if not terms:
            return {}

        n_terms = len(terms)

        if n_terms == 1:
            return {0: 0.5 - box_width / 2}

        total_box_width = n_terms * box_width
        available_space = layer_width - total_box_width

        if n_terms > 1:
            spacing = available_space / (n_terms - 1)
        else:
            spacing = 0

        min_spacing = 0.015
        if spacing < min_spacing and n_terms > 1:
            spacing = min_spacing
            box_width_adjusted = (layer_width - spacing * (n_terms - 1)) / n_terms
        else:
            box_width_adjusted = box_width

        total_used_width = n_terms * box_width_adjusted + (n_terms - 1) * spacing
        start_x_centered = (1.0 - total_used_width) / 2

        positions = {}
        for i in range(n_terms):
            positions[i] = start_x_centered + i * (box_width_adjusted + spacing)

        return positions

    def _sort_terms_by_robustness(self, terms: List[Dict]) -> List[Dict]:
        """Sort terms by robustness (descending - highest first)"""
        return sorted(terms, key=lambda t: t.get('frequency_percentage', 0), reverse=True)

    def create_simple_visualization(self, data: Dict, output_path: str, title: str = "Hierarchical Manifold Organization", subtitle: str = "GO Biological Process Terms", dpi: int = 300, figsize: Tuple[int, int] = (20, 10), robustness_threshold: float = 0.0):
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        Y_INTEGRATION_CONTEXT_BOUNDARY = 0.6
        Y_CORE_INTEGRATION_BOUNDARY = 0.4
        Y_CENTER_ZERO_LINE = 0.5

        box_height = 0.06
        box_width_default = 0.16
        font_size_terms = 14
        font_size_delta = 13
        wrap_max_width = 22

        # Continuous color regions (Core top, Context bottom)
        ax.add_patch(plt.Rectangle((0, 0), 1, Y_CORE_INTEGRATION_BOUNDARY,
                                   facecolor=self.color_specific, edgecolor='none', zorder=0))
        ax.add_patch(plt.Rectangle((0, Y_CORE_INTEGRATION_BOUNDARY), 1,
                                   Y_INTEGRATION_CONTEXT_BOUNDARY - Y_CORE_INTEGRATION_BOUNDARY,
                                   facecolor=self.color_bridge, edgecolor='none', zorder=0))
        ax.add_patch(plt.Rectangle((0, Y_INTEGRATION_CONTEXT_BOUNDARY), 1,
                                   1.0 - Y_INTEGRATION_CONTEXT_BOUNDARY,
                                   facecolor=self.color_general, edgecolor='none', zorder=0))

        # Vertical labels on the left (Core top, Context bottom)
        label_x = 0.02
        ax.text(label_x, Y_CORE_INTEGRATION_BOUNDARY / 2, 'Core Function Layer\n($\\alpha > \\beta$)',
                fontsize=14, fontweight='bold', rotation=90, va='center', ha='center',
                color=self.edge_color_specific)
        ax.text(label_x, Y_CENTER_ZERO_LINE, 'Integration Layer\n($\\alpha \\approx \\beta$)',
                fontsize=14, fontweight='bold', rotation=90, va='center', ha='center',
                color=self.edge_color_bridge)
        ax.text(label_x, Y_INTEGRATION_CONTEXT_BOUNDARY + (1.0 - Y_INTEGRATION_CONTEXT_BOUNDARY) / 2,
                'Context Layer\n($\\beta > \\alpha$)',
                fontsize=14, fontweight='bold', rotation=90, va='center', ha='center',
                color=self.edge_color_general)

        # Separation lines and subtle boundary markers
        ax.axhline(y=Y_CORE_INTEGRATION_BOUNDARY, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=1)
        ax.axhline(y=Y_INTEGRATION_CONTEXT_BOUNDARY, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=1)

        # Subtle boundary markers on the right side
        line_x_right = 0.99
        line_length = 0.015

        ax.plot([line_x_right - line_length, line_x_right], [1.0, 1.0],
                color='black', linewidth=1.2, alpha=0.6, zorder=2)
        ax.text(line_x_right - line_length - 0.01, 1.0, '1', va='center', ha='right',
                fontsize=8, color='black', alpha=0.6)

        ax.plot([line_x_right - line_length, line_x_right], [Y_INTEGRATION_CONTEXT_BOUNDARY, Y_INTEGRATION_CONTEXT_BOUNDARY],
                color='black', linewidth=1.2, alpha=0.6, zorder=2)
        ax.text(line_x_right - line_length - 0.01, Y_INTEGRATION_CONTEXT_BOUNDARY, '0.2', va='center', ha='right',
                fontsize=8, color='black', alpha=0.6)

        ax.plot([line_x_right - line_length, line_x_right], [Y_CENTER_ZERO_LINE, Y_CENTER_ZERO_LINE],
                color='black', linewidth=1.2, alpha=0.6, zorder=2)
        ax.text(line_x_right - line_length - 0.01, Y_CENTER_ZERO_LINE, '0', va='center', ha='right',
                fontsize=8, color='black', alpha=0.6)

        ax.plot([line_x_right - line_length, line_x_right], [Y_CORE_INTEGRATION_BOUNDARY, Y_CORE_INTEGRATION_BOUNDARY],
                color='black', linewidth=1.2, alpha=0.6, zorder=2)
        ax.text(line_x_right - line_length - 0.01, Y_CORE_INTEGRATION_BOUNDARY, '-0.2', va='center', ha='right',
                fontsize=8, color='black', alpha=0.6)

        ax.plot([line_x_right - line_length, line_x_right], [0.0, 0.0],
                color='black', linewidth=1.2, alpha=0.6, zorder=2)
        ax.text(line_x_right - line_length - 0.01, 0.0, '-1', va='center', ha='right',
                fontsize=8, color='black', alpha=0.6)

        # Axes for RS and Delta

        origin_x = 0.99
        origin_y = 0.10

        vertical_up = 0.07
        vertical_down = 0.07
        horizontal_left = 0.03

        arrow_vert = dict(
            arrowstyle="<->",
            lw=1.8,
            color="black",
            mutation_scale=10,
            shrinkA=0, shrinkB=0
        )

        arrow_horiz = dict(
            arrowstyle="-|>",
            lw=1.8,
            color="black",
            mutation_scale=10,
            shrinkA=0, shrinkB=0
        )

        ax.annotate("",
                    xy=(origin_x, origin_y + vertical_up),
                    xytext=(origin_x, origin_y - vertical_down),
                    arrowprops=arrow_vert)

        ax.annotate("",
                    xy=(origin_x - horizontal_left, origin_y),
                    xytext=(origin_x, origin_y),
                    arrowprops=arrow_horiz)

        ax.text(origin_x + 0.004, origin_y,
                r"$\Delta$",
                ha="left", va="center",
                fontsize=12, fontweight="bold")

        ax.text(origin_x - horizontal_left - 0.003, origin_y,
                "RS",
                ha="right", va="center",
                fontsize=12, fontweight="bold")

        ax.axis("off")

        beta_terms = self._extract_individual_terms_with_diff(data['beta_greater']['groups'])
        equal_terms = self._extract_individual_terms_with_diff(data['equal']['groups'])
        alpha_terms = self._extract_individual_terms_with_diff(data['alpha_greater']['groups'])

        if robustness_threshold > 0:
            beta_terms = [t for t in beta_terms if t.get('frequency_percentage', 0) >= robustness_threshold]
            equal_terms = [t for t in equal_terms if t.get('frequency_percentage', 0) >= robustness_threshold]
            alpha_terms = [t for t in alpha_terms if t.get('frequency_percentage', 0) >= robustness_threshold]

        group_color_beta = self.edge_color_general
        if beta_terms:
            beta_terms_sorted = self._sort_terms_by_robustness(beta_terms)

            x_positions = self._calculate_x_positions_ranked(beta_terms_sorted,
                                                             start_x=0.08,
                                                             layer_width=0.84,
                                                             box_width=box_width_default)

            for i, term in enumerate(beta_terms_sorted):
                diff_norm = term['difference_normalized']
                robustness = term.get('frequency_percentage', 0)

                x_pos = x_positions[i]

                y_center = self._calculate_y_position_from_diff(diff_norm, Y_CENTER_ZERO_LINE)
                y_pos = y_center - box_height / 2

                term_color = group_color_beta

                wrapped_text = self._wrap_text(term['name'], max_width=wrap_max_width)
                ax.text(x_pos + box_width_default / 2, y_pos + box_height / 2, wrapped_text,
                        ha='center', va='center',
                        fontsize=font_size_terms,
                        color=term_color, fontweight='bold', zorder=11,
                        linespacing=0.85)

                diff_text = f"Δ={diff_norm:.2f}, RS={robustness:.1f}"
                ax.text(x_pos + box_width_default / 2, y_pos - 0.012, diff_text,
                        ha='center', va='top',
                        fontsize=font_size_delta,
                        color=term_color, style='italic', zorder=11)

        group_color_equal = self.edge_color_bridge
        if equal_terms:
            equal_terms_sorted = self._sort_terms_by_robustness(equal_terms)

            x_positions = self._calculate_x_positions_ranked(equal_terms_sorted,
                                                             start_x=0.08,
                                                             layer_width=0.84,
                                                             box_width=box_width_default)

            for i, term in enumerate(equal_terms_sorted):
                diff_norm = term['difference_normalized']
                robustness = term.get('frequency_percentage', 0)

                x_pos = x_positions[i]

                y_center = self._calculate_y_position_from_diff(diff_norm, Y_CENTER_ZERO_LINE)
                y_pos = y_center - box_height / 2

                term_color = group_color_equal

                wrapped_text = self._wrap_text(term['name'], max_width=wrap_max_width)
                ax.text(x_pos + box_width_default / 2, y_pos + box_height / 2, wrapped_text,
                        ha='center', va='center',
                        fontsize=font_size_terms,
                        color=term_color, fontweight='bold', zorder=11,
                        linespacing=0.85)

                diff_text = f"Δ={diff_norm:.2f}, RS={robustness:.1f}"
                ax.text(x_pos + box_width_default / 2, y_pos - 0.012, diff_text,
                        ha='center', va='top',
                        fontsize=font_size_delta,
                        color=term_color, style='italic', zorder=11)

        group_color_alpha = self.edge_color_specific
        if alpha_terms:
            alpha_terms_sorted = self._sort_terms_by_robustness(alpha_terms)

            x_positions = self._calculate_x_positions_ranked(alpha_terms_sorted,
                                                             start_x=0.08,
                                                             layer_width=0.84,
                                                             box_width=box_width_default)

            for i, term in enumerate(alpha_terms_sorted):
                diff_norm = term['difference_normalized']
                robustness = term.get('frequency_percentage', 0)

                x_pos = x_positions[i]

                y_center = self._calculate_y_position_from_diff(diff_norm, Y_CENTER_ZERO_LINE)
                y_pos = y_center - box_height / 2

                term_color = group_color_alpha

                wrapped_text = self._wrap_text(term['name'], max_width=wrap_max_width)
                ax.text(x_pos + box_width_default / 2, y_pos + box_height / 2, wrapped_text,
                        ha='center', va='center',
                        fontsize=font_size_terms,
                        color=term_color, fontweight='bold', zorder=11,
                        linespacing=0.85)
                diff_text = f"Δ={diff_norm:.2f}, RS={robustness:.1f}"
                ax.text(x_pos + box_width_default / 2, y_pos - 0.012, diff_text,
                        ha='center', va='top',
                        fontsize=font_size_delta,
                        color=term_color, style='italic', zorder=11)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()


def run_single_visualization(ontology_dir: Path, robustness_threshold: float = 0.0):
    """Runs visualization for a single, specific ontology path."""
    visualizer = HierarchicalManifoldVisualizer()

    ontology_name = ontology_dir.name
    try:
        signature_name = ontology_dir.parent.parent.name
    except IndexError:
        print(f"Error: The path '{ontology_dir}' is too short. It must follow the structure 'SIGNATURE/parameter_analysis/ONTOLOGY'.")
        sys.exit(1)

    manifold_file = ontology_dir / f"manifold_analysis_{ontology_name}.json"

    if not manifold_file.exists():
        print(f"Error: Manifold file not found: {manifold_file}")
        sys.exit(1)

    output_dir_final = ontology_dir / OUTPUT_SUBDIR
    output_dir_final.mkdir(parents=True, exist_ok=True)

    output_base_name = f"manifold_{ontology_name}"
    simple_output_path = output_dir_final / f"{output_base_name}_simple.png"

    try:
        data = visualizer.load_manifold_data(str(manifold_file))

        title = f"Hierarchical Manifold Organization of {signature_name.replace('_genes', '').replace('_', ' ')}"
        subtitle = f"GO {ontology_name} Terms - Organized by Specificity Gradient"

        visualizer.create_simple_visualization(
            data=data,
            output_path=str(simple_output_path),
            title=title,
            subtitle=subtitle,
            robustness_threshold=robustness_threshold
        )

    except Exception as e:
        print(f"Critical error during visualization: {e}")
        raise


def process_all_signatures(input_dir: Path, robustness_threshold: float = 0.0):
    """Traverses the input directory and processes all signatures and ontologies."""

    data_subpath = Path("parameter_analysis")
    signature_dirs = [
        d for d in input_dir.iterdir()
        if d.is_dir() and (d / data_subpath).exists()
    ]

    if not signature_dirs:
        print(f"Error: No signature directories found under '{input_dir.resolve()}' containing '{data_subpath}'")
        sys.exit(1)

    visualizer = HierarchicalManifoldVisualizer()

    progress_bar = tqdm(signature_dirs, desc="Signatures", unit="sig", dynamic_ncols=True)

    for signature_dir in progress_bar:
        signature_base_path = signature_dir / data_subpath

        ontology_dirs = [
            d for d in signature_base_path.iterdir()
            if d.is_dir() and (d / f"manifold_analysis_{d.name}.json").exists()
        ]

        if not ontology_dirs:
            continue

        for ontology_dir in ontology_dirs:
            ontology_name = ontology_dir.name
            manifold_file = ontology_dir / f"manifold_analysis_{ontology_name}.json"

            if not manifold_file.exists():
                continue

            output_dir_final = ontology_dir / OUTPUT_SUBDIR
            output_dir_final.mkdir(parents=True, exist_ok=True)

            output_base_name = f"manifold_{ontology_name}"
            simple_output_path = output_dir_final / f"{output_base_name}_simple.png"

            try:
                data = visualizer.load_manifold_data(str(manifold_file))

                title = f"Hierarchical Manifold Organization of {signature_dir.name.replace('_genes', '').replace('_', ' ')}"
                subtitle = f"GO {ontology_name} Terms - Organized by Specificity Gradient"

                visualizer.create_simple_visualization(
                    data=data,
                    output_path=str(simple_output_path),
                    title=title,
                    subtitle=subtitle,
                    robustness_threshold=robustness_threshold
                )

            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch visualization of Hierarchical GO Term Manifolds."
    )
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--input_dir",
        type=str,
        help="Root directory for batch processing."
    )

    group.add_argument(
        "--specific_path",
        type=str,
        help="Specific path to a single ontology folder."
    )

    parser.add_argument(
        "--robustness",
        type=float,
        default=20.0,
        help="Minimum robustness threshold (frequency_percentage)."
    )

    args = parser.parse_args()

    if args.input_dir:
        input_path = Path(args.input_dir).resolve()
        if not input_path.is_dir():
            print(f"Error: The specified input directory does not exist: {input_path}")
            sys.exit(1)
        process_all_signatures(input_path, args.robustness)

    elif args.specific_path:
        specific_path = Path(args.specific_path).resolve()
        if not specific_path.is_dir():
            print(f"Error: The specified path does not exist: {specific_path}")
            sys.exit(1)
        run_single_visualization(specific_path, args.robustness)