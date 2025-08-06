# Trial Data Aggregation

This module provides functionality to aggregate trial data from Bayesian optimization experiments into Polars DataFrames for analysis.

## Overview

The trial aggregation system is designed to work with the following directory structure:

```
base_path/
├── 0/                    # Trial 0
│   ├── experiment_snapshot.json
│   ├── trial_results.json
│   ├── session_data.parquet
│   └── other_metadata.json
├── 1/                    # Trial 1
│   ├── experiment_snapshot.json
│   └── ...
├── 2/                    # Trial 2
│   └── ...
└── ...
```

## Main Functions

### `aggregate_trial_data(base_path, experiment_name=None)`

Aggregates trial data from directories into a Polars DataFrame.

**Parameters:**
- `base_path` (str): Base file path containing trial directories
- `experiment_name` (str, optional): Name of the experiment to filter for specific experiment files

**Returns:**
- `pl.DataFrame`: DataFrame containing aggregated trial data with columns:
  - `trial_index`: Trial number
  - `parameters`: Dictionary of trial parameters
  - `reward`: Reward observation value
  - `metadata`: Trial metadata dictionary
  - `experiment_snapshot`: Full experiment snapshot
  - `local_path`: Path to trial directory
  - `files_present`: List of files found in trial directory

### `extract_trial_parameters(df)`

Extracts trial parameters into separate columns for easier analysis.

**Parameters:**
- `df` (pl.DataFrame): DataFrame from `aggregate_trial_data()`

**Returns:**
- `pl.DataFrame`: DataFrame with parameters expanded into separate columns (prefixed with `param_`)

### `get_trial_summary(df)`

Generates a summary of the trial data.

**Parameters:**
- `df` (pl.DataFrame): DataFrame from `aggregate_trial_data()`

**Returns:**
- `Dict[str, Any]`: Summary statistics and information about the trials

### `find_experiment_directories(base_path, pattern="*")`

Finds experiment directories that contain trial data.

**Parameters:**
- `base_path` (str): Base path to search for experiment directories
- `pattern` (str): Pattern to match directory names

**Returns:**
- `List[str]`: List of paths to experiment directories containing trial data

### `aggregate_multiple_experiments(base_path, pattern="*")`

Aggregates trial data from multiple experiments.

**Parameters:**
- `base_path` (str): Base path to search for experiment directories
- `pattern` (str): Pattern to match experiment directory names

**Returns:**
- `Dict[str, pl.DataFrame]`: Dictionary mapping experiment names to their trial DataFrames

## Usage Examples

### Basic Usage

```python
from src.analysis.aggregate_trial_data import aggregate_trial_data, get_trial_summary

# Aggregate trial data
df = aggregate_trial_data("/path/to/your/experiment/directory")

# Get summary statistics
summary = get_trial_summary(df)
print(f"Total trials: {summary['total_trials']}")
print(f"Trials with rewards: {summary['trials_with_rewards']}")
```

### Extract Parameters

```python
from src.analysis.aggregate_trial_data import extract_trial_parameters

# Extract parameters into separate columns
df_expanded = extract_trial_parameters(df)

# Now you can access parameters directly
print(df_expanded.select(['trial_index', 'param_amplitude', 'param_frequency']))
```

### Multiple Experiments

```python
from src.analysis.aggregate_trial_data import aggregate_multiple_experiments

# Aggregate multiple experiments
experiments = aggregate_multiple_experiments("/path/to/experiments/base/directory")

# Access individual experiments
for exp_name, exp_df in experiments.items():
    print(f"Experiment {exp_name}: {len(exp_df)} trials")
```

### Save and Load Data

```python
from src.analysis.aggregate_trial_data import save_aggregated_data, load_aggregated_data

# Save aggregated data
save_aggregated_data(df, "my_trials.parquet")

# Load saved data
loaded_df = load_aggregated_data("my_trials.parquet")
```

## File Types Supported

The aggregation system looks for and processes the following file types:

1. **`experiment_snapshot.json`**: Contains trial parameters and results from the Ax experiment
2. **`trial_results.json`**: Contains trial-specific results and metadata
3. **`*.parquet`**: Session data files (basic metadata is extracted)
4. **Other JSON files**: Additional metadata files

## Expected Data Structure

### Experiment Snapshot JSON Structure

```json
{
  "trials": [
    {
      "index": 0,
      "arm": {
        "parameters": {
          "amplitude": 1.5,
          "frequency": 130.0
        }
      },
      "data": {
        "by_metric_name": {
          "bilaterally_weighted_average_NREM_delta_power": {
            "mean": 0.75
          }
        }
      }
    }
  ]
}
```

### Trial Results JSON Structure

```json
{
  "bilaterally_weighted_average_NREM_delta_power": 0.75,
  "metadata": {
    "session_info": "...",
    "evaluation_time": "..."
  }
}
```

## Error Handling

The system includes robust error handling:

- Missing files are logged but don't stop processing
- Invalid JSON files are skipped with error messages
- Corrupted parquet files are handled gracefully
- Empty directories are reported

## Performance Considerations

- The system processes files sequentially to avoid memory issues
- Large parquet files are only read for metadata (not loaded entirely)
- JSON files are parsed once and cached in the DataFrame

## Integration with Existing Codebase

This module integrates with the existing Bayesian optimization infrastructure:

- Works with the `ExperimentTracker` class output
- Compatible with the `Reporter` class file structure
- Supports the Ax experiment format used throughout the codebase
- Uses Polars for efficient data manipulation (following project conventions)

## Troubleshooting

### No trials found
- Check that the base path contains numeric subdirectories
- Verify that trial directories contain experiment files
- Ensure file permissions allow reading the directories

### Missing reward data
- Check that `experiment_snapshot.json` contains trial data
- Verify that reward metrics are named correctly
- Look for `trial_results.json` as an alternative source

### Parameter extraction issues
- Ensure parameters are stored as dictionaries in the JSON files
- Check that parameter names are consistent across trials
- Verify JSON structure matches expected format 