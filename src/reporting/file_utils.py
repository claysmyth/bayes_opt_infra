import os
import zipfile
import fnmatch
import subprocess
import csv
import shutil
import shutil
import glob

# Variables
CSV_COLUMNS = [
    "project",
    "time_stamp",
    "device",
    "experiment",
    "info",
    "run_name",
    "WandB_url",
    "WandB_id",
    "run_dir",
    "date",
    "time",
    "commit",
    "commit_branch",
]


def zipdir(path, ziph, exclude_patterns=[]):
    """
    Zip an entire directory, optionally excluding some subdirectories and files based on patterns.

    :param path: Path to directory to zip
    :param ziph: ZipFile handle
    :param exclude_patterns: List of patterns to exclude
    """
    for root, dirs, files in os.walk(path):
        # Exclude directories that match any of the exclude_patterns
        dirs[:] = [
            d
            for d in dirs
            if not any(fnmatch.fnmatch(d, pattern) for pattern in exclude_patterns)
        ]
        for file in files:
            if not any(fnmatch.fnmatch(file, pattern) for pattern in exclude_patterns):
                ziph.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), path),
                )


def create_zip(source_directory, output_filename, exclude=[]):
    """
    Create a zip file from a directory, excluding specified patterns.

    :param source_directory: Directory to zip
    :param output_filename: Name of the output zip file
    :param exclude: List of patterns to exclude
    """
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipdir(source_directory, zipf, exclude_patterns=exclude)


def get_git_info():
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return {"git_commit": commit, "git_branch": branch}
    except subprocess.CalledProcessError:
        print("An error occurred while trying to fetch git info")
        return None


def dict_to_csv(data_dict, csv_file_path):
    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file in write mode, create if not exists
    with open(csv_file_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=data_dict.keys())

        # If the file does not exist, write the header first
        if not file_exists:
            writer.writeheader()

        # Write the dictionary as a new row in the CSV
        writer.writerow(data_dict)


def add_config_to_csv(config, csv):
    # Create a dictionary from the config
    config_dict = {k: v for k, v in config.items() if k in CSV_COLUMNS}

    config_dict["date"], config_dict["time"] = config_dict["time_stamp"].split("_")

    # Add the config to the CSV
    dict_to_csv(config_dict, csv)


def save_conda_package_versions(run_dir):
    """
    Print the list of conda packages and their versions to the console.
    """
    # Command to get the list of packages and their versions
    command = ["conda", "list"]

    # Execute the command and capture the output
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)

    # Write the output to a text file
    with open(os.path.join(run_dir, "conda_packages.txt"), "w") as file:
        file.write(result.stdout)


def ensure_parent_directory_exists(file_path):
    """
    This function checks if the parent directory of the given file path exists.
    If it does not exist, it creates the parent directory.
    """
    import os

    # Extract the parent directory from the file path
    parent_dir = os.path.dirname(file_path)

    # Check if the parent directory exists
    if not os.path.exists(parent_dir):
        # Create the parent directory if it does not exist
        os.makedirs(parent_dir)
        print(f"Created directory: {parent_dir}")
    else:
        # print(f"Directory already exists: {parent_dir}")
        return


def copy_file_to_matching_subdirs(directory_path, file_name, pattern):
    """
    Copies a file into all subdirectories of a given directory that match a specified glob pattern.

    :param directory_path: Path of the directory
    :param file_name: Name of the file to copy
    :param pattern: The glob pattern to match subdirectory names
    """
    # Full path of the file to be copied
    if not directory_path in file_name:
        file_path = os.path.join(directory_path, file_name)
    else:
        file_path = file_name

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"The file {file_name} does not exist in the directory {directory_path}")
        return

    # Construct the full pattern path
    full_pattern = os.path.join(directory_path, pattern)

    # Iterate over all directories matching the glob pattern
    for dir_path in glob.glob(full_pattern):
        if os.path.isdir(dir_path):
            # Copy the file to this subdirectory
            shutil.copy(file_path, dir_path)
            print(f"Copied {file_name} to {dir_path}")
