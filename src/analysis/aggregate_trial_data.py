import polars as pl
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import glob


def aggregate_trial_data(base_path: str, experiment_name: str = None) -> pl.DataFrame:
    """
    Aggregate trial data from directories into a Polars table.
    
    This function cycles through trial directories (numbered by trial index) and aggregates
    the trial metadata, parameters, and reward observations into a single Polars DataFrame.
    
    Parameters
    ----------
    base_path : str
        Base file path containing trial directories
    experiment_name : str, optional
        Name of the experiment to filter for specific experiment files
        
    Returns
    -------
    pl.DataFrame
        DataFrame containing aggregated trial data with columns:
        - trial_index: Trial number
        - parameters: Dictionary of trial parameters
        - reward: Reward observation value
        - metadata: Trial metadata dictionary
        - experiment_snapshot: Full experiment snapshot
        - local_path: Path to trial directory
        - files_present: List of files found in trial directory
    """
    
    # Get all subdirectories that are numeric (trial indices)
    trial_dirs = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.isdigit():
            trial_dirs.append((int(item), item_path))
    
    # Sort by trial index
    trial_dirs.sort(key=lambda x: x[0])
    
    # Initialize lists to store data
    trial_data = []
    
    for trial_index, trial_path in trial_dirs:
        trial_info = {
            'trial_index': trial_index,
            'local_path': trial_path,
            'files_present': [],
            'parameters': None,
            'reward': None,
            'metadata': None,
            'experiment_snapshot': None
        }
        
        # List all files in the trial directory
        files = os.listdir(trial_path)
        trial_info['files_present'] = files
        
        # Look for specific file types and load data
        for file in files:
            file_path = os.path.join(trial_path, file)
            
            # Load experiment snapshot (contains trial parameters and results)
            if file == 'experiment_snapshot.json':
                try:
                    with open(file_path, 'r') as f:
                        experiment_data = json.load(f)
                    trial_info['experiment_snapshot'] = experiment_data
                    
                    # Extract trial-specific information
                    if 'trials' in experiment_data:
                        # Find the specific trial in the experiment
                        for trial in experiment_data['trials']:
                            if trial.get('index') == trial_index:
                                trial_info['parameters'] = trial.get('arm', {}).get('parameters', {})
                                # Extract reward from trial results
                                if 'data' in trial and 'by_metric_name' in trial['data']:
                                    # Look for reward metrics (common names)
                                    reward_metrics = ['bilaterally_weighted_average_NREM_delta_power', 
                                                    'reward', 'objective', 'cost']
                                    for metric in reward_metrics:
                                        if metric in trial['data']['by_metric_name']:
                                            trial_info['reward'] = trial['data']['by_metric_name'][metric]['mean']
                                            break
                                break
                except Exception as e:
                    print(f"Error loading experiment_snapshot.json for trial {trial_index}: {e}")
            
            # Load trial results (if saved separately)
            elif file == 'trial_results.json':
                try:
                    with open(file_path, 'r') as f:
                        results_data = json.load(f)
                    trial_info['metadata'] = results_data
                    
                    # Extract reward if not already found
                    if trial_info['reward'] is None:
                        reward_metrics = ['bilaterally_weighted_average_NREM_delta_power', 
                                        'reward', 'objective', 'cost']
                        for metric in reward_metrics:
                            if metric in results_data:
                                trial_info['reward'] = results_data[metric]
                                break
                except Exception as e:
                    print(f"Error loading trial_results.json for trial {trial_index}: {e}")
            
            # Load parquet files (session data)
            elif file.endswith('.parquet'):
                try:
                    # Read parquet file to get basic info
                    df = pl.read_parquet(file_path)
                    trial_info['metadata'] = {
                        'parquet_file': file,
                        'rows': len(df),
                        'columns': df.columns,
                        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
                    }
                except Exception as e:
                    print(f"Error reading parquet file {file} for trial {trial_index}: {e}")
            
            # Load other JSON files that might contain metadata
            elif file.endswith('.json') and file not in ['experiment_snapshot.json', 'trial_results.json']:
                try:
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                    if trial_info['metadata'] is None:
                        trial_info['metadata'] = {}
                    trial_info['metadata'][file.replace('.json', '')] = json_data
                except Exception as e:
                    print(f"Error loading {file} for trial {trial_index}: {e}")
        
        trial_data.append(trial_info)
    
    # Convert to Polars DataFrame
    df = pl.DataFrame(trial_data)
    
    return df


def extract_trial_parameters(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract trial parameters into separate columns.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame from aggregate_trial_data()
        
    Returns
    -------
    pl.DataFrame
        DataFrame with parameters expanded into separate columns
    """
    
    # Get unique parameter names across all trials
    all_params = set()
    for params in df['parameters'].drop_nulls():
        if isinstance(params, dict):
            all_params.update(params.keys())
    
    # Create new columns for each parameter
    for param_name in sorted(all_params):
        df = df.with_columns(
            pl.col('parameters').map_elements(
                lambda x: x.get(param_name) if isinstance(x, dict) else None
            ).alias(f'param_{param_name}')
        )
    
    return df


def get_trial_summary(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of the trial data.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame from aggregate_trial_data()
        
    Returns
    -------
    Dict[str, Any]
        Summary statistics and information about the trials
    """
    
    summary = {
        'total_trials': len(df),
        'trials_with_rewards': len(df.filter(pl.col('reward').is_not_null())),
        'trials_with_parameters': len(df.filter(pl.col('parameters').is_not_null())),
        'trials_with_experiment_snapshot': len(df.filter(pl.col('experiment_snapshot').is_not_null())),
        'trial_indices': df['trial_index'].to_list(),
        'reward_stats': None,
        'parameter_names': set()
    }
    
    # Get reward statistics
    rewards = df['reward'].drop_nulls()
    if len(rewards) > 0:
        summary['reward_stats'] = {
            'mean': rewards.mean(),
            'std': rewards.std(),
            'min': rewards.min(),
            'max': rewards.max(),
            'count': len(rewards)
        }
    
    # Get parameter names
    for params in df['parameters'].drop_nulls():
        if isinstance(params, dict):
            summary['parameter_names'].update(params.keys())
    
    summary['parameter_names'] = sorted(list(summary['parameter_names']))
    
    return summary


def find_experiment_directories(base_path: str, pattern: str = "*") -> List[str]:
    """
    Find experiment directories that contain trial data.
    
    Parameters
    ----------
    base_path : str
        Base path to search for experiment directories
    pattern : str
        Pattern to match directory names (default: "*")
        
    Returns
    -------
    List[str]
        List of paths to experiment directories containing trial data
    """
    
    experiment_dirs = []
    
    # Search for directories that contain numeric subdirectories (trial indices)
    for root, dirs, files in os.walk(base_path):
        # Check if any subdirectories are numeric (potential trial indices)
        numeric_dirs = [d for d in dirs if d.isdigit()]
        if numeric_dirs:
            # This directory contains trial data
            experiment_dirs.append(root)
    
    # Filter by pattern if specified
    if pattern != "*":
        experiment_dirs = [d for d in experiment_dirs if pattern in os.path.basename(d)]
    
    return experiment_dirs


def aggregate_multiple_experiments(base_path: str, pattern: str = "*") -> Dict[str, pl.DataFrame]:
    """
    Aggregate trial data from multiple experiments.
    
    Parameters
    ----------
    base_path : str
        Base path to search for experiment directories
    pattern : str
        Pattern to match experiment directory names
        
    Returns
    -------
    Dict[str, pl.DataFrame]
        Dictionary mapping experiment names to their trial DataFrames
    """
    
    experiment_dirs = find_experiment_directories(base_path, pattern)
    experiments = {}
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        try:
            df = aggregate_trial_data(exp_dir)
            if len(df) > 0:  # Only include experiments with trial data
                experiments[exp_name] = df
        except Exception as e:
            print(f"Error processing experiment {exp_name}: {e}")
    
    return experiments


# Example usage and utility functions
def save_aggregated_data(df: pl.DataFrame, output_path: str) -> None:
    """
    Save aggregated trial data to parquet file.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to save
    output_path : str
        Path to save the parquet file
    """
    df.write_parquet(output_path)
    print(f"Saved aggregated trial data to {output_path}")


def load_aggregated_data(input_path: str) -> pl.DataFrame:
    """
    Load aggregated trial data from parquet file.
    
    Parameters
    ----------
    input_path : str
        Path to the parquet file
        
    Returns
    -------
    pl.DataFrame
        Loaded trial data
    """
    return pl.read_parquet(input_path) 