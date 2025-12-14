# Building a Reproducible Machine Learning Experiment Pipeline

## Project Description

This project demonstrates best practices for building reproducible ML pipelines, including configuration management, experiment logging, model versioning, and result tracking. The focus is on creating a framework that ensures experiments can be easily reproduced, compared, and shared. Reproducibility is fundamental to scientific research and practical ML, and this project provides a comprehensive template for building experiment pipelines.

**Why This Project Matters**:
- **Reproducibility**: Ensures experiments can be exactly reproduced, which is critical for scientific validity and production deployment
- **Experiment Management**: Provides systematic approach to managing multiple experiments and their results
- **Version Control**: Tracks model versions, configurations, and results for better organization
- **Collaboration**: Makes it easy to share and compare experiments across team members
- **Best Practices**: Demonstrates industry-standard practices for ML experiment management
- **Production Readiness**: Builds habits and infrastructure needed for production ML systems

**Key Research Questions**:
- How can we ensure ML experiments are fully reproducible?
- What infrastructure is needed for systematic experiment management?
- How do we track and compare multiple experiment runs?
- What metadata should be captured for each experiment?
- How can we make experiment comparison and analysis easier?

**Pipeline Features**:
- **Configuration Management**: YAML-based configuration files for all hyperparameters
- **Experiment Logging**: Automatic tracking of metrics, parameters, and results
- **Model Versioning**: Save and load models with version tracking
- **Reproducibility**: Fixed random seeds, environment tracking
- **Result Comparison**: Compare multiple experiment runs side-by-side
- **Metadata Capture**: Track dataset versions, code versions, and system information

## Dataset Description

**Dataset Name**: California Housing Dataset

**Source**: Scikit-learn datasets (originally from 1990 US Census)

**Dataset Details**:
- **Number of samples**: 20,640 housing districts from California
- **Number of features**: 8 numerical features
  - **MedInc**: Median income in block group
  - **HouseAge**: Median house age in block group
  - **AveRooms**: Average number of rooms per household
  - **AveBedrms**: Average number of bedrooms per household
  - **Population**: Block group population
  - **AveOccup**: Average number of household members
  - **Latitude**: Block group latitude
  - **Longitude**: Block group longitude
- **Target variable**: Median house value (continuous, in hundreds of thousands of dollars)
- **Task**: Regression (predicting median house value)
- **Data quality**: Clean dataset with no missing values

**Why This Dataset**:
- **Well-understood problem**: Classic regression task perfect for demonstrating pipeline concepts without dataset complexity
- **Moderate size**: Large enough to be realistic but small enough for quick iteration during pipeline development
- **Clean data**: No missing values, minimal preprocessing needed, allowing focus on pipeline infrastructure
- **Standard benchmark**: Widely used dataset allows for meaningful comparisons and validation
- **Quick experiments**: Fast to load and process, enabling rapid pipeline testing and iteration

**Data Loading**:
```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X, y = housing.data, housing.target
```

**IMPORTANT**: No synthetic or hard-coded data is used in this project. All experiments use the real California Housing dataset loaded from scikit-learn's `fetch_california_housing()` function.

## Project Structure

```
project5_experiment_pipeline/
├── README.md
├── requirements.txt
├── configs/
│   └── experiment_config.yaml
├── notebooks/
│   ├── 01_pipeline_design.ipynb
│   ├── 02_experiment_execution.ipynb
│   └── 03_reproducibility_demo.ipynb
└── src/
    ├── pipeline.py
    ├── config_loader.py
    └── experiment_logger.py
```

## Key Features

1. **Configuration Management**: 
   - YAML-based configs for all hyperparameters
   - Hierarchical configuration structure
   - Environment-specific configs

2. **Experiment Logging**: 
   - Track all experiments with unique IDs
   - Log metrics, parameters, and results
   - Timestamp and version tracking

3. **Model Versioning**: 
   - Save and load models with metadata
   - Version tracking and comparison
   - Model artifact management

4. **Reproducibility**: 
   - Fixed random seeds
   - Code version tracking (git)
   - Environment information capture

5. **Result Comparison**: 
   - Compare multiple runs
   - Visualize experiment history
   - Export results for analysis

## Learning Objectives

1. Design ML experiment pipelines
2. Implement configuration management
3. Create experiment logging systems
4. Ensure reproducibility
5. Compare experiment results
6. Build reusable experiment frameworks

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Configure experiments in `configs/experiment_config.yaml`
3. Open Jupyter notebooks in order (01 → 03)
4. Run experiments and review results in `experiments/` directory

## Requirements

- Python 3.8+
- NumPy, Pandas
- Scikit-learn
- PyYAML
- Matplotlib, Seaborn
- Jupyter
