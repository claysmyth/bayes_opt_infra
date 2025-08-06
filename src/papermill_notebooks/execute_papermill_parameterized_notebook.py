import sys
import json
import yaml
import papermill as pm
from pathlib import Path
# Add current directory to path
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Map notebook enum to actual notebook file paths
NOTEBOOK_MAP = {
    "summary_stats": "participant_summary_stats_for_bayes_reward_parameterized.ipynb",
    "nrem_model": "NREM_model_check_and_adaptive_config_gen_parameterized.ipynb",
    "setup_baseline": "setup_baseline_duckdb_parameterized.ipynb",
    # Add more mappings as needed
}

def load_config(config_path):
    config_path = Path(config_path)
    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported config file type. Use .json or .yaml")

def main():
    """
    Execute a parameterized Jupyter notebook using papermill. 
    Probably should run from sleepclass3 environment.

    Usage:
        python papermill_notebook_execution.py <notebook> <config_file_path> [output_path]

    Arguments:
        <notebook>         Enum-like string to select which notebook to run (see NOTEBOOK_MAP).
        <config_file_path> Path to the participant-specific config file (JSON or YAML).
        [output_path]      (Optional) Path to save the executed notebook. If not provided, defaults to the config file's directory with the name <notebook>_output.ipynb.
    """
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python papermill_notebook_execution.py <notebook> <config_file_path> [output_path]")
        sys.exit(1)

    notebook_enum = sys.argv[1]
    config_file_path = sys.argv[2]
    output_path = None
    if len(sys.argv) == 4:
        output_path = sys.argv[3]

    # Select notebook path
    if notebook_enum not in NOTEBOOK_MAP:
        print(f"Unknown notebook: {notebook_enum}. Available: {list(NOTEBOOK_MAP.keys())}")
        sys.exit(1)
    notebook_path = Path(current_dir) / NOTEBOOK_MAP[notebook_enum]

    # Load config
    parameters = load_config(config_file_path)

    # Set output notebook path
    if output_path is None:
        output_path = str(Path(config_file_path).parent / f"{notebook_enum}_output.ipynb")
    else:
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / f"{notebook_enum}_output.ipynb"
        output_path = str(output_path)

    # Execute notebook
    pm.execute_notebook(
        notebook_path,
        output_path,
        parameters=parameters
    )
    print(f"Executed {notebook_path} with parameters from {config_file_path}. Output: {output_path}")

if __name__ == "__main__":
    main()