"""
Enhanced Configuration Manager for Neural Network Project
Provides centralized configuration management with validation and defaults
"""

import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

class ConfigManager:
    """Enhanced configuration manager with validation and environment support"""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with fallback to defaults"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                print(f"Configuration loaded from {self.config_path}")
                return config
            else:
                print(f"Config file {self.config_path} not found.")
                return {}
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return {}
    
    def _setup_logging(self):
        """Setup logging based on config"""
        log_level = self.get('logging.level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _validate_config(self):
        """Validate essential configuration parameters"""
        required_paths = [
            'data.csv_path',
            'data.loading.train_ratio',
            'data.loading.val_ratio', 
            'data.loading.test_ratio',
            'model.training.epochs',
            'model.architecture.hidden_layers'
        ]
        
        # Add prediction mode specific validation
        mode = self.get('pipeline.mode', 'train')
        if mode == 'prediction':
            prediction_required = [
                'prediction.model_path',
                'prediction.target_year',
                'prediction.sequence_length'
            ]
            required_paths.extend(prediction_required)
        
        for path in required_paths:
            if self.get(path) is None:
                self.logger.warning(f"Missing required config parameter: {path}")
        
        # Validate ratios sum to 1 (only for train/grid_search modes)
        if mode in ['train', 'grid_search']:
            ratios = [
                self.get('data.loading.train_ratio', 0.7),
                self.get('data.loading.val_ratio', 0.15),
                self.get('data.loading.test_ratio', 0.15)
            ]
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError(f"Data split ratios must sum to 1.0, got {sum(ratios)}")
        
        # Set defaults for missing data plot configuration if not present
        if self.get('output.plots.missing_data') is None:
            self.set('output.plots.missing_data', True)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get('model.training.epochs')
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config_section = self.config
        
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        config_section[keys[-1]] = value
    
    def get_data_path(self) -> Path:
        """Get absolute path to data file"""
        csv_path = self.get('data.csv_path')
        if not os.path.isabs(csv_path):
            # Make relative to project root (go up two levels from src/config/)
            project_root = Path(__file__).parent.parent.parent
            return project_root / csv_path
        return Path(csv_path)
    
    def get_output_dir(self) -> Path:
        """Create and return output directory path"""
        output_dir = Path(self.get('output.results_dir', 'data/output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file"""
        save_path = Path(output_path) if output_path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
        self.logger.info(f"Configuration saved to {save_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()
    
    def update_from_dict(self, update_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, update_dict)
        self._validate_config()

# Global config instance
config = ConfigManager()