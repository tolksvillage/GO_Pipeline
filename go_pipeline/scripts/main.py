import os
import subprocess
import sys
import argparse

sys.path.insert(1,'.')

from go_pipeline.scripts.helper.pipeline_state import PipelineState

def run_step(step_name: str, command: list[str], state: PipelineState,
             state_keys: list[str] = None, verbose: bool = True) -> None:
    """
    state_keys: the checkpoint key(s) this invocation covers.

    A step is skipped only if ALL of its state_keys are already marked done.
    On success, ALL of its state_keys are marked done.

    This is what allows a single 'both' run to satisfy two independently
    tracked sub-modes (fixed + cumulative) in one invocation, while a later
    run that only needs one of them (e.g. just 'cumulative') can correctly
    see that the 'fixed' part is already covered and skip just that part,
    regardless of which mode was used in earlier runs and in which order.
    """
    keys = state_keys if state_keys else [step_name]
    state.reload()

    if all(state.is_done("steps", k) for k in keys):
        print(f"\n=== Skip (already processed): {step_name} ===")
        return

    print(f"\n=== Starting: {step_name} ===")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    if verbose:
        result = subprocess.run(command, env=env)
    else:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    state.reload()

    if result.returncode != 0:
        if not verbose:
            stdout_text = result.stdout.decode("utf-8", errors="replace")
            stderr_text = result.stderr.decode("utf-8", errors="replace")
            if stdout_text.strip():
                print(stdout_text)
            if stderr_text.strip():
                print(stderr_text)
        for k in keys:
            state.mark_failed("steps", k, f"returncode={result.returncode}")
        raise RuntimeError(f"Step failed: {step_name}")

    for k in keys:
        state.mark_done("steps", k)


def get_dilution_submodes(with_dilution_analysis: bool, dilution_mode) -> list[str]:
    """
    Decomposes the requested run configuration into the individual dilution
    sub-modes it is made of, so each one can be checkpointed independently:

      - no dilution at all          -> ["no_dilution"]
      - --dilution_mode=fixed       -> ["fixed"]
      - --dilution_mode=cumulative  -> ["cumulative"]
      - --dilution_mode=both        -> ["fixed", "cumulative"]

    This is the key to order-independence: a 'both' run is just "fixed AND
    cumulative", each tracked under its own key. Whichever of the two was
    already completed in an earlier run (in any order) is recognized as
    done and skipped; only the missing one actually runs.
    """
    if not with_dilution_analysis:
        return ["no_dilution"]
    if dilution_mode == "both":
        return ["fixed", "cumulative"]
    return [dilution_mode]


def ask_yes_no(question: str) -> bool:
    while True:
        answer = input(question).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")

def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Please use true or false.")


def ask_choice(question: str, options: list[str]) -> str:
    print(question)
    for i, option in enumerate(options, start=1):
        print(f"  [{i}] {option}")
    while True:
        answer = input(f"Choice (1-{len(options)}): ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        print(f"Please enter a number between 1 and {len(options)}.")


def main() -> None:
    python_executable = sys.executable

    parser = argparse.ArgumentParser(
        description="Run GO pipeline",
    )

    parser.add_argument("--output_path", type=str, help="Path for output results")
    parser.add_argument("--signatures_path", type=str, help="Path for signatures")

    parser.add_argument(
        "--with_dilution",
        action="store_true",
        help="Enable dilution analysis"
    )

    parser.add_argument(
        "--dilution_mode",
        choices=["cumulative", "fixed", "both"],
        help="Dilution mode"
    )

    parser.add_argument(
        "--with_llm_request",
        type=str_to_bool,
        default=None,
        help="Set to true or false to enable or disable GO term definition, NCBI summaries, and LLM analysis"
    )

    parser.add_argument(
        "--with_paths",
        type=str_to_bool,
        default=False,
        help="Run path collector and path rankings (true/false)"
    )

    args = parser.parse_args()

    if not (args.output_path or args.signatures_path):
        print("\nNo paths were provided via command-line arguments.")
        print("Please enter the required paths manually.")
        print("Tip: You can skip prompts next time by running:")
        print(
            f"  python {os.path.basename(__file__)} "
            "--output_path=<PATH> --signatures_path=<PATH>\n"
        )

    if args.signatures_path:
        signatures_path = args.signatures_path
    else:
        signatures_path = input(
            "Please enter the signatures path (directory containing files like signatures1.txt, signatures2.txt, ...): "
        ).strip()
        while not signatures_path:
            signatures_path = input(
                "Path cannot be empty. Please enter the signatures path: "
            ).strip()

    if args.output_path:
        output_path = args.output_path
    else:
        output_path = input(
            "Please enter the output path (directory where results should be stored): "
        ).strip()
        while not output_path:
            output_path = input(
                "Path cannot be empty. Please enter the output path: "
            ).strip()

    if args.with_dilution and args.dilution_mode:
        with_dilution_analysis = True
        dilution_mode = args.dilution_mode
    else:
        print("\nNo complete dilution settings were provided via command-line arguments.")
        print("Please specify whether dilution analysis should be performed and, if yes, which mode to use.")
        print("Tip: You can skip these prompts next time by running:")
        print(
            f"  python {os.path.basename(__file__)} "
            "--with_dilution --dilution_mode=<cumulative|fixed|both>\n"
        )

        with_dilution_analysis = ask_yes_no("Should dilution analysis be performed? (y/n) ")

        dilution_mode = None
        if with_dilution_analysis:
            dilution_mode = ask_choice(
                "Dilution mode:",
                ["cumulative", "fixed", "both"]
            )

    if args.with_llm_request is not None:
        with_llm_request = args.with_llm_request
    else:
        print("\nNo LLM request setting was provided via command-line arguments.")
        print(
            "Please specify whether GO term definition, NCBI summary annotation, and LLM analysis should be performed.")
        print("Tip: You can skip this prompt next time by running:")
        print(
            f"  python {os.path.basename(__file__)} "
            "--with_llm_request=true\n"
        )

        with_llm_request = ask_yes_no(
            "Should GO term definition, NCBI summary annotation, and LLM analysis be performed? (y/n) "
        )

    with_paths = args.with_paths

    submodes = get_dilution_submodes(with_dilution_analysis, dilution_mode)

    state_file = os.path.join(output_path, ".pipeline_state.json")
    state = PipelineState(state_file)

    steps = []

    if with_dilution_analysis:
        steps.append(
            (
                "Create diluted signatures...",
                [
                    python_executable,
                    "-m",
                    "go_pipeline.scripts.dilute_signatures",
                    f"--signatures={signatures_path}",
                    f"--gaf=data/goa_human.gaf",
                    f"--genes=data/all_human_genes/all_genes.txt",
                    "--steps=10",
                    f"--mode={dilution_mode}",
                    f"--output={signatures_path}",
                    f"--state_file={state_file}",
                ],
                [f"Create diluted signatures...::{sm}" for sm in submodes],
            )
        )

    shared_step_specs = [
        (
            "Mapping genes to GO-terms...",
            [
                python_executable,
                "-m",
                "go_pipeline.scripts.genes_to_ont",
                f"--base_path={signatures_path}",
                f"--output_path={output_path}",
                f"--state_file={state_file}",
            ],
        ),
        (
            "Reduce terms by representatives...",
            [
                python_executable,
                "-m",
                "go_pipeline.scripts.representatives",
                f"--input_dir={output_path}",
                f"--state_file={state_file}",
            ],
        ),
        (
            "Extract most enriched terms I...",
            [
                python_executable,
                "-m",
                "go_pipeline.scripts.paths_keywords_from_representatives",
                f"--input={output_path}",
                f"--output={output_path}",
                f"--signatures={signatures_path}",
                f"--state_file={state_file}",
            ],
        ),
        (
            "Extract most enriched terms II...",
            [
                python_executable,
                "-m",
                "go_pipeline.scripts.parameter_analysis_main",
                f"--input_path={output_path}",
                f"--state_file={state_file}",
            ],
        ),
        (
            "Visualize most enriched terms I...",
            [
                python_executable,
                "-m",
                "go_pipeline.scripts.parameter_analysis_divide_manifold",
                f"--input_path={output_path}",
                f"--state_file={state_file}",
            ],
        ),
        (
            "Visualize most enriched terms II...",
            [
                python_executable,
                "-m",
                "go_pipeline.scripts.manifold_visualizer",
                f"--input_dir={output_path}",
                f"--state_file={state_file}",
            ],
        ),
        (
            "Add GO term definitions and gene symbols...",
            [
                python_executable,
                "-m",
                "go_pipeline.scripts.llm_request.get_term_definition",
                f"--input_data={output_path}",
                "--obo_file=data/go-basic.obo",
                f"--state_file={state_file}",
            ],
        ),
        (
            "Add NCBI gene summaries...",
            [
                python_executable,
                "-m",
                "go_pipeline.scripts.llm_request.get_NCBI_infos",
                f"--input_data={output_path}",
                "--gene_summary=data/NCBI/gene_summary.gz",
                "--gene_info=data/Homo_sapiens.gene_info.gz",
                f"--state_file={state_file}",
            ],
        ),
    ]

    # These steps read/write the whole signatures directory at once (not one
    # call per sub-mode), but which sub-modes' diluted files already exist on
    # disk can change between runs. So their checkpoint must cover every
    # sub-mode that is part of *this* run; if any one of them is still
    # missing, the step has to run again (the scripts themselves then skip
    # whatever was already processed internally, file by file).
    for name, command in shared_step_specs:
        steps.append((name, command, [f"{name}::{sm}" for sm in submodes]))

    if with_dilution_analysis:
        if dilution_mode in ("fixed", "both"):
            steps.append(
                (
                    "Dilution Analysis for fixed mode...",
                    [
                        python_executable,
                        "-m",
                        "go_pipeline.scripts.dilute_analysis",
                        f"--input_path={output_path}",
                        f"--signatures={output_path}",
                        "--mode=fixed",
                        "--analyze_all",
                        "--auto_cutoff",
                        f"--state_file={state_file}",
                    ],
                    ["Dilution Analysis for fixed mode...::fixed"],
                )
            )
            steps.append(
                (
                    "Saving results in Excel...",
                    [
                        python_executable,
                        "-m",
                        "go_pipeline.scripts.create_summary_data",
                        f"--input_path={output_path}",
                        "--mode", "fixed",
                        "--ontology", "all",
                        f"--state_file={state_file}",

                    ],
                    ["Saving results in Excel...::fixed"],
                )
            )
        if dilution_mode in ("cumulative", "both"):
            steps.append(
                (
                    "Dilution Analysis for cumulative mode...",
                    [
                        python_executable,
                        "-m",
                        "go_pipeline.scripts.dilute_analysis",
                        f"--input_path={output_path}",
                        f"--signatures={output_path}",
                        "--mode=cumulative",
                        "--analyze_all",
                        "--auto_cutoff",
                        f"--state_file={state_file}",
                    ],
                    ["Dilution Analysis for cumulative mode...::cumulative"],
                )
            )
            # Note: intentionally no Excel export here, same as the original
            # script - create_summary_data is only ever run for fixed mode.

    if with_paths:
        steps.extend([
            (
                "Collect GO paths...",
                [
                    python_executable,
                    "-m",
                    "go_pipeline.scripts.paths.path_collector",
                    f"--base_dir={output_path}",
                    f"--state_file={state_file}",
                ],
                ["Collect GO paths..."],
            ),
            (
                "Rank GO paths...",
                [
                    python_executable,
                    "-m",
                    "go_pipeline.scripts.paths.path_rankings",
                    f"--input_dir={output_path}",
                    f"--state_file={state_file}",
                ],
                ["Rank GO paths..."],
            ),
        ])

    if with_llm_request:
        llm_command = [
            python_executable,
            "-m",
            "go_pipeline.scripts.llm_request.llm_request",
            f"--input_dir={output_path}",
            f"--state_file={state_file}",
        ]

        if with_dilution_analysis and dilution_mode in ("fixed", "both"):
            llm_command.append("--filtered")

        steps.append(
            (
                "Run LLM signature analysis...",
                llm_command,
                ["Run LLM signature analysis..."],
            )
        )

    for step_name, command, state_keys in steps:
        run_step(step_name, command, state, state_keys)

    print("\nFinished.")


if __name__ == "__main__":
    main()