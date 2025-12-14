"""
Machine Learning Experiment Pipeline

This module implements a reproducible ML pipeline with configuration
management, logging, and model versioning.
"""

import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import joblib


class MLPipeline:
    """
    Reproducible ML experiment pipeline.
    """
    
    def __init__(self, config: Dict[str, Any], experiment_name: str = None):
        """
        Initialize pipeline.
        
        Parameters:
        -----------
        config : Dict
            Experiment configuration
        experiment_name : str
            Name for this experiment
        """
        self.config = config
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {}
        self.model = None
        
        # Create experiment directory
        self.exp_dir = Path(f"experiments/{self.experiment_name}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(config.get('random_seed', 42))
    
    def save_config(self):
        """Save experiment configuration."""
        config_path = self.exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric."""
        if name not in self.results:
            self.results[name] = []
        self.results[name].append({'value': value, 'step': step})
    
    def save_model(self, model: Any, filename: str = "model.pkl"):
        """Save trained model."""
        model_path = self.exp_dir / filename
        joblib.dump(model, model_path)
        self.model = model
    
    def load_model(self, filename: str = "model.pkl"):
        """Load saved model."""
        model_path = self.exp_dir / filename
        self.model = joblib.load(model_path)
        return self.model
    
    def save_results(self):
        """Save experiment results."""
        results_path = self.exp_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        return {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }

