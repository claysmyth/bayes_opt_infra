# bayes_opt_infra
Infrastructure to run bayesian optimization on neural data.

This project uses Bayesian Optimization to determine neural stimulation parameters for participants to optimize some clinically or scientifically relevant outcome. It draws on data from a local server. 

It uses config files to initialize sub-pipeline objects, and then uses these objects to run the Bayesian Optimization pipeline. Config files contain the parameters and functions to run for each sub-pipeline object.

The general workflow is as follows:

1. Global config is loaded from a yaml file. This holds the configuration for all sub-pipeline objects and high-level file paths.

2. Pipeline objects are initialized:
   - Reporter: Manages visualizations and reports, handling both local files and W&B logging
   - Experiment Tracker: Manages Bayes optimization experiments and parameter updates
   - Session Manager: Tracks reported/unreported sessions via Project CSV
   - Data Source: Retrieves session data from storage
   - Quality AC (optional): Performs quality assurance and control checks
   - Evaluation: Calculates objective values from session data

3. For each new session:
   a. Get experiment ID and update experimental context
   b. Retrieve recent session data from data source (which presumably corresponds to the last bayesian optimization trial)
   c. If Quality AC is enabled:
      - Run quality assurance checks (prevents bad data from entering optimization)
      - Run quality control checks (holds insufficient data until complete)
   d. Calculate objective value(s) using evaluation function
   e. Update Bayesian optimizer with results using Ax
   f. Get next parameters to try
   g. Save and ship new parameters to destination (e.g., RC+S device)
   h. Generate visualizations and reports
   i. Mark session as reported in session manager

This pipeline relies heavily on the 'Ax' library, which abstracts much of the Bayesian Optimization process using Botorch: [text](https://ax.dev/versions/0.1.9/)

The pipeline is designed to be:
- Modular: Each component has a specific responsibility
- Configurable: All behaviors defined in YAML configs
- Extensible: New components can be added via config
- Robust: Includes quality checks and error handling
- Traceable: Full logging and visualization support

## Setup
[Installation instructions here]

## Configuration
[Configuration details here]

## Usage
[Usage instructions here]

