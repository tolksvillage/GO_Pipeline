import os
import subprocess
import sys
import argparse

def run_step(step_name: str, command: list[str], verbose: bool = True) -> None:
    print(f"\n=== Starting: {step_name} ===")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    if verbose:
        result = subprocess.run(command, env=env)
    else:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

    if result.returncode != 0:
        if not verbose:
            stdout_text = result.stdout.decode("utf-8", errors="replace")
            stderr_text = result.stderr.decode("utf-8", errors="replace")

            if stdout_text.strip():
                print(stdout_text)
            if stderr_text.strip():
                print(stderr_text)

        raise RuntimeError(f"Step failed: {step_name}")


def ask_yes_no(question: str) -> bool:
    while True:
        answer = input(question).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


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
                ],
            )
        )

    steps.extend([
        (
            "Mapping genes to GO-terms...",
            [
                python_executable,
                "-m",
                "go_pipeline.scripts.genes_to_ont",
                f"--base_path={signatures_path}",
                f"--output_path={output_path}",
            ],
        ),
        (
            "Reduce terms by representatives...",
            [
                python_executable,
                "-m",
                "go_pipeline.scripts.representatives",
                f"--input_dir={output_path}",
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
            ],
        ),
        (
            "Extract most enriched terms II...",
            [
                python_executable,
                "-m",
                "go_pipeline.scripts.parameter_analysis_main",
                f"--input_path={output_path}",
            ],
        ),
        (
            "Visualize most enriched terms I...",
            [
                python_executable,
                "-m",
                "go_pipeline.scripts.parameter_analysis_divide_manifold",
                f"--input_path={output_path}",
            ],
        ),
        (
            "Visualize most enriched terms II...",
            [
                python_executable,
                "-m",
                "go_pipeline.scripts.manifold_visualizer",
                f"--input_dir={output_path}",
            ],
        ),
    ])

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
                    ],
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
                    ],
                )
            )

    for step_name, command in steps:
        run_step(step_name, command)

    print("\nFinished.")


if __name__ == "__main__":
    main()