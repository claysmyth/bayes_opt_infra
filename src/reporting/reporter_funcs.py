import os
from .file_utils import (
    create_zip,
    get_git_info,
    save_conda_package_versions,
)
import altair, plotly, matplotlib, seaborn, wandb
from prefect import get_run_logger
from prefect.artifacts import create_markdown_artifact
import polars as pl


def local_setup(path, config, conda=False):

    # Save code, git info, and config file to run directory
    create_zip(
        f"{os.getcwd()}/python",
        f"{path}/code.zip",
        exclude=config["code_snapshot_exclude"],
    )
    if (
        conda
    ):  # Don't save conda package versions, because we are trying to run from non-conda terminal.
        save_conda_package_versions(path)
    git_info = get_git_info()
    # Write git info to a text file
    git_info_path = os.path.join(path, "git_info.txt")
    with open(git_info_path, "w") as f:
        for key, value in git_info.items():
            f.write(f"{key}: {value}\n")


def log_plotting_result(result, func_name, log_options, wandb_run=None, path=None):

    logger = get_run_logger()

    logging_actions = {
        altair.Chart: _log_html_plot,
        plotly.graph_objs.Figure: _log_plotly_plot,
        # wandb.plot.line: _log_wandb_line,
        (matplotlib.figure.Figure, seaborn.axisgrid.FacetGrid): _log_image_plot,
        wandb.Table: _log_wandb_table,
        dict: _log_dict,
        pl.DataFrame: _log_polars_table,
        # (tuple, list): _log_many, # For logging multiple plots, each one likely a dict for wandb
    }

    for types, action in logging_actions.items():
        if isinstance(result, types):
            action(result, func_name, log_options, wandb_run, path)
            return

    if isinstance(result, str) and result.startswith("<!DOCTYPE html>"):
        _log_html_string(result, func_name, log_options, wandb_run, path)
    else:
        logger.warning(f"Unsupported result type for {func_name}: {type(result)}")


# Helper functions for logging different types of results
def _log_html_plot(result, func_name, log_options, wandb_run, path):
    html_path = os.path.join(path, f"{func_name}.html")
    result.save(html_path)
    table = wandb.Table(columns=["Altair Plot"])
    table.add_data(wandb.Html(html_path))
    _log_to_wandb(wandb_run, func_name, table, log_options)


def _log_plotly_plot(result, func_name, log_options, wandb_run, path):
    _log_to_wandb(wandb_run, func_name, result, log_options)
    file_path = _log_to_file(path, func_name, result.to_html(), ".html", log_options)


def _log_wandb_line(result, func_name, log_options, wandb_run, path):
    _log_to_wandb(wandb_run, func_name, result, log_options)


def _log_image_plot(result, func_name, log_options, wandb_run, path):
    file_path = _log_to_file(
        path, func_name, result, ".png", log_options, save_func=result.savefig
    )
    _log_to_wandb(wandb_run, func_name, wandb.Image(result), log_options)


def _log_wandb_table(result, func_name, log_options, wandb_run, path):
    file_path = _log_to_file(
        path, func_name, result, ".csv", log_options, save_func=result.to_csv
    )
    _log_to_wandb(wandb_run, func_name, result, log_options)


def _log_html_string(result, func_name, log_options, wandb_run, path):
    file_path = _log_to_file(path, func_name, result, ".html", log_options)
    _log_to_wandb(wandb_run, func_name, wandb.Html(result), log_options)


def _log_to_wandb(wandb_run, func_name, content, log_options):
    if "WandB" in log_options and wandb_run:
        wandb_run.log({func_name: content})


def _log_dict(result, func_name, log_options, wandb_run, path):
    if "WandB" in log_options and wandb_run:
        if len(result) == 1:
            wandb_run.log(result)
        else:
            {wandb_run.log({key: value}) for key, value in result.items()}
    if "file" in log_options and path:
        file_path = os.path.join(path, f"{func_name}.json")
        with open(file_path, "w") as f:
            f.write(result)


def _log_polars_table(result, func_name, log_options, wandb_run, path):
    if "WandB" in log_options and wandb_run:
        wandb_run.log({func_name: result.to_pandas()})
    if "file" in log_options and path:
        _log_to_file(
            path,
            func_name,
            result,
            ".parquet",
            log_options,
            save_func=result.write_parquet,
        )


def _log_to_file(path, func_name, content, extension, log_options, save_func=None):
    if ("file" in log_options) and path:
        file_path = os.path.join(path, f"{func_name}{extension}")
        if save_func:
            save_func(file_path)
        else:
            with open(file_path, "w") as f:
                f.write(content)
        if "prefect" in log_options:
            create_markdown_artifact(
                key=f"{func_name}_{'plot' if extension in ['.html', '.png'] else 'table'}",
                markdown=f"{'Plot' if extension in ['.html', '.png'] else 'Table'} saved as {extension[1:].upper()}: {file_path}",
                description=f"{extension[1:].upper()} {'plot' if extension in ['.html', '.png'] else 'table'} for {func_name}",
            )
        return file_path
    return None
