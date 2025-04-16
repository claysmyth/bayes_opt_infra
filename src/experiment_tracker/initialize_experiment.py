import yaml
import argparse
from pathlib import Path
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.storage.sqa_store.db import (
    init_engine_and_session_factory,
    get_engine,
    create_all_tables,
)
from ax.storage.sqa_store.structs import DBSettings


def load_config(config_path: str) -> dict:
    """Load configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def default_ax_setup(config: dict) -> AxClient:
    """Initialize default Ax experiment with configuration."""
    # Extract parameters from config
    base_dir = Path(config["paths"]["base_dir"])
    experiment_name = config["experiment"]["name"]

    # Ensure base directory exists
    base_dir.mkdir(parents=True, exist_ok=True)

    # Initialize AxClient with or without database
    ax_client_kwargs = {"random_seed": config["experiment"].get("random_seed", 42)}

    if config["experiment"].get("use_database", False):
        # Initialize database
        db_path = base_dir / f"{experiment_name}.db"
        url = f"sqlite:///{db_path}"

        # Setup database if requested
        if config["experiment"].get("setup_database", False):
            init_engine_and_session_factory(url=url)
            engine = get_engine()
            create_all_tables(engine)

        # Add database settings to AxClient
        ax_client_kwargs["db_settings"] = DBSettings(url=url)

    ax_client = AxClient(**ax_client_kwargs)

    # Create experiment
    ax_client.create_experiment(
        name=experiment_name,
        # parameters=[
        #     {
        #         "name": param["name"],
        #         "type": param["type"],
        #         "bounds": param["bounds"],
        #         "value_type": param.get("value_type", "float"),
        #         "log_scale": param.get("log_scale", False),
        #     }
        #     for param in config["parameters"]
        # ],
        parameters=config["parameters"],
        objectives={
            obj["name"]: ObjectiveProperties(minimize=obj.get("minimize", True))
            for obj in config["objectives"]
        },
        choose_generation_strategy_kwargs=config.get(
            "choose_generation_strategy_kwargs", {}
        ),
    )

    # Save ax_client to JSON
    json_path = base_dir / f"{experiment_name}.json"
    ax_client.save_to_json_file(
        filepath=str(json_path),
    )

    return ax_client


def main():
    parser = argparse.ArgumentParser(description="Initialize Ax experiment from config")
    parser.add_argument("config_path", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load config and initialize experiment
    config = load_config(args.config_path)
    ax_client = default_ax_setup(config)

    # Save initial state to JSON (optional)
    if config["experiment"].get("save_json", True):
        json_path = (
            Path(config["paths"]["base_dir"]) / f"{config['experiment']['name']}.json"
        )
        ax_client.save_to_json_file(filepath=str(json_path))

    print(f"Experiment '{config['experiment']['name']}' initialized successfully!")


if __name__ == "__main__":
    # Run with python initialize_experiment.py [filepath to config yaml. See template in configs/ax_experiment_init_config/experiment_initialization_config.yaml]
    main()
