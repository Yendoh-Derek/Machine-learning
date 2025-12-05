"""
Utility functions for path management and data operations.
"""
import os
import json
from pathlib import Path


class PathManager:
    """Centralized path management for the project."""
    
    def __init__(self, base_dir: str):
        self.BASE_DIR = Path(base_dir)
        
        # Data directories
        self.RAW_DIR = self.BASE_DIR / "data" / "raw"
        self.PROCESSED_DIR = self.BASE_DIR / "data" / "processed"
        self.CACHE_DIR = self.BASE_DIR / "data" / "cache"
        
        # Artifacts directories
        self.PREPROCESSING_DIR = self.BASE_DIR / "artifacts" / "preprocessing"
        self.RL_MODELS_DIR = self.BASE_DIR / "artifacts" / "rl_models"
        self.ENV_DIR = self.BASE_DIR / "artifacts" / "env"
        self.INDEXES_DIR = self.BASE_DIR / "artifacts" / "indexes"
        
        # Other directories
        self.EXPORTS_DIR = self.BASE_DIR / "exports"
        self.NOTEBOOKS_DIR = self.BASE_DIR / "notebooks"
        
        # Create all directories
        self._create_dirs()
    
    def _create_dirs(self):
        """Create all required directories if they don't exist."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Path) and attr_name.endswith('_DIR'):
                attr.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, *args) -> Path:
        """Construct a path relative to BASE_DIR."""
        return self.BASE_DIR.joinpath(*args)


def save_json(data: dict, filepath: Path):
    """Save dictionary to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> dict:
    """Load JSON file to dictionary."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_dataframe(df, required_columns: list, name: str = "DataFrame"):
    """
    Validate that a DataFrame contains required columns and check basic integrity.
    
    Args:
        df: pandas DataFrame
        required_columns: list of column names that must exist
        name: name for logging purposes
    
    Returns:
        dict with validation results
    """
    results = {
        'valid': True,
        'missing_columns': [],
        'null_counts': {},
        'shape': df.shape
    }
    
    # Check for missing columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        results['valid'] = False
        results['missing_columns'] = list(missing)
    
    # Count nulls for each column
    results['null_counts'] = df[required_columns].isnull().sum().to_dict()
    
    print(f"\n{'='*60}")
    print(f"Validation Report: {name}")
    print(f"{'='*60}")
    print(f"Shape: {results['shape'][0]} rows × {results['shape'][1]} columns")
    print(f"Valid: {results['valid']}")
    
    if results['missing_columns']:
        print(f"\n⚠️  Missing Columns: {results['missing_columns']}")
    
    print(f"\nNull Counts:")
    for col, count in results['null_counts'].items():
        pct = (count / len(df)) * 100
        print(f"  {col}: {count} ({pct:.2f}%)")
    
    return results