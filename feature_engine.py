"""
Feature engineering utilities for LLM recommendation system.
Includes text preprocessing, cost scoring, and feature normalization.
"""
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pickle
from pathlib import Path


class TextPreprocessor:
    """
    Text preprocessing utilities for model descriptions and tags.
    """
    
    @staticmethod
    def clean_description(text: Optional[str]) -> str:
        """
        Clean and normalize model description text.
        
        Args:
            text: Raw description text
            
        Returns:
            Cleaned text
        """
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Lowercase
        text = text.lower().strip()
        
        return text
    
    @staticmethod
    def parse_tags(tags: str) -> List[str]:
        """
        Parse tags from string representation back to list.
        
        Args:
            tags: String representation of tags list
            
        Returns:
            List of cleaned tags
        """
        if not tags or pd.isna(tags):
            return []
        
        # Handle string representation of list
        if isinstance(tags, str):
            # Remove brackets and quotes
            tags = tags.strip('[]').replace("'", "").replace('"', '')
            # Split by comma
            tag_list = [t.strip() for t in tags.split(',') if t.strip()]
        else:
            tag_list = []
        
        return tag_list
    
    @staticmethod
    def combine_text_features(
        description: str,
        tags: List[str],
        use_cases: Optional[str] = None
    ) -> str:
        """
        Combine multiple text fields into single string for embedding.
        
        Args:
            description: Model description
            tags: List of tags
            use_cases: Optional use cases text
            
        Returns:
            Combined text string
        """
        parts = []
        
        if description:
            parts.append(description)
        
        if tags:
            # Join tags with spaces
            tag_str = " ".join(tags)
            parts.append(tag_str)
        
        if use_cases and not pd.isna(use_cases):
            parts.append(str(use_cases))
        
        return " ".join(parts)


class CostScorer:
    """
    Calculate cost scores based on model size and quantization.
    Lower score = lower cost (better for efficiency).
    """
    
    # Quantization cost multipliers (relative to FP16)
    QUANT_MULTIPLIERS = {
        'FP16': 1.0,      # Baseline
        'FP8': 0.5,
        'INT8': 0.5,
        '8-bit': 0.5,
        'INT4': 0.25,
        '4-bit': 0.25,
        'GPTQ': 0.3,      # Typically 4-bit
        'AWQ': 0.3,       # Typically 4-bit
        '2-bit': 0.125,
    }
    
    @classmethod
    def calculate_cost_score(
        cls,
        num_parameters: Optional[float],
        quantization: Optional[str]
    ) -> float:
        """
        Calculate cost score for a model.
        
        Cost = base_cost * quantization_multiplier
        
        Args:
            num_parameters: Number of parameters in billions
            quantization: Quantization type
            
        Returns:
            Cost score (higher = more expensive to run)
        """
        # Default to 7B if unknown (median LLM size)
        base_cost = num_parameters if num_parameters else 7.0
        
        # Get quantization multiplier
        if quantization and quantization in cls.QUANT_MULTIPLIERS:
            quant_mult = cls.QUANT_MULTIPLIERS[quantization]
        else:
            quant_mult = 1.0  # Assume FP16 if unknown
        
        return base_cost * quant_mult
    
    @classmethod
    def normalize_cost_score(cls, cost_scores: pd.Series) -> pd.Series:
        """
        Normalize cost scores to [0, 1] range using min-max scaling.
        
        Args:
            cost_scores: Series of raw cost scores
            
        Returns:
            Normalized scores (0 = cheapest, 1 = most expensive)
        """
        min_cost = cost_scores.min()
        max_cost = cost_scores.max()
        
        if max_cost == min_cost:
            return pd.Series([0.5] * len(cost_scores), index=cost_scores.index)
        
        return (cost_scores - min_cost) / (max_cost - min_cost)


class PopularityScorer:
    """
    Calculate popularity scores from downloads and likes.
    """
    
    @staticmethod
    def calculate_popularity_score(
        downloads: int,
        likes: int,
        download_weight: float = 0.7,
        like_weight: float = 0.3
    ) -> float:
        """
        Calculate popularity score with logarithmic scaling.
        
        Score = download_weight * log(downloads + 1) + like_weight * log(likes + 1)
        
        Args:
            downloads: Number of downloads
            likes: Number of likes
            download_weight: Weight for downloads
            like_weight: Weight for likes
            
        Returns:
            Popularity score
        """
        download_score = np.log1p(downloads)
        like_score = np.log1p(likes)
        
        return download_weight * download_score + like_weight * like_score
    
    @staticmethod
    def normalize_popularity(popularity_scores: pd.Series) -> pd.Series:
        """
        Normalize popularity scores to [0, 1] range.
        
        Args:
            popularity_scores: Series of raw popularity scores
            
        Returns:
            Normalized scores
        """
        min_score = popularity_scores.min()
        max_score = popularity_scores.max()
        
        if max_score == min_score:
            return pd.Series([0.5] * len(popularity_scores), index=popularity_scores.index)
        
        return (popularity_scores - min_score) / (max_score - min_score)


class RecencyScorer:
    """
    Calculate recency scores based on last modification date.
    """
    
    @staticmethod
    def calculate_recency_score(
        created_at: Optional[str],
        reference_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate recency score (days since creation).
        Does NOT cap at 5 years - returns actual age.
        
        Args:
            created_at: ISO format timestamp
            reference_date: Reference date (defaults to now)
            
        Returns:
            Days since creation (actual value, no capping)
        """
        if not created_at or pd.isna(created_at):
            return None  # Return None instead of default
        
        if reference_date is None:
            reference_date = pd.Timestamp.utcnow()
        
        try:
            created = pd.to_datetime(created_at)
            days_old = (reference_date - created).days
            
            # Handle future dates (API errors)
            if days_old < 0:
                return 0  # Treat as brand new
            
            return float(days_old)  # Return actual age, no capping
        except:
            return None
    
    @staticmethod
    def normalize_recency(recency_scores: pd.Series, use_actual_max: bool = True) -> pd.Series:
        """
        Normalize recency scores to [0, 1] range.
        
        Args:
            recency_scores: Series of raw recency scores (days old)
            use_actual_max: If True, use actual max in data. If False, use 5 years.
            
        Returns:
            Normalized scores (0 = oldest, 1 = newest)
        """
        # Handle None/NaN values - fill with median
        valid_scores = recency_scores.dropna()
        
        if len(valid_scores) == 0:
            return pd.Series([0.5] * len(recency_scores), index=recency_scores.index)
        
        # Determine max for normalization
        if use_actual_max:
            max_days = valid_scores.max()
        else:
            max_days = 1825.0  # 5 years fixed cap
        
        min_days = valid_scores.min()
        
        # Handle edge case where all models same age
        if max_days == min_days:
            return pd.Series([0.5] * len(recency_scores), index=recency_scores.index)
        
        # Normalize: newer = higher score (invert the scale)
        normalized = recency_scores.fillna(valid_scores.median())
        normalized = 1.0 - ((normalized - min_days) / (max_days - min_days))
        
        return normalized


class FeatureScaler:
    """
    Scaler for feature normalization with save/load capability.
    """
    
    def __init__(self):
        self.feature_stats = {}
    
    def fit(self, df: pd.DataFrame, columns: List[str]):
        """
        Fit scaler to data.
        
        Args:
            df: DataFrame with features
            columns: List of column names to scale
        """
        for col in columns:
            if col in df.columns:
                self.feature_stats[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted statistics.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        df_scaled = df.copy()
        
        for col, stats in self.feature_stats.items():
            if col in df_scaled.columns:
                # Min-max scaling
                min_val = stats['min']
                max_val = stats['max']
                
                if max_val != min_val:
                    df_scaled[col] = (df_scaled[col] - min_val) / (max_val - min_val)
                else:
                    df_scaled[col] = 0.5
        
        return df_scaled
    
    def save(self, filepath: Path):
        """Save scaler to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.feature_stats, f)
    
    def load(self, filepath: Path):
        """Load scaler from disk."""
        with open(filepath, 'rb') as f:
            self.feature_stats = pickle.load(f)


class DiversityCalculator:
    """
    Calculate diversity features for models.
    """
    
    @staticmethod
    def calculate_architecture_diversity(
        model_type: Optional[str],
        architecture_counts: Dict[str, int]
    ) -> float:
        """
        Calculate diversity score based on architecture rarity.
        Rarer architectures get higher diversity scores.
        
        Args:
            model_type: Architecture type
            architecture_counts: Dictionary of architecture -> count
            
        Returns:
            Diversity score (0-1, higher = more diverse/rare)
        """
        if not model_type or pd.isna(model_type):
            return 0.5
        
        total_models = sum(architecture_counts.values())
        arch_count = architecture_counts.get(model_type, 1)
        
        # Inverse frequency
        diversity = 1.0 - (arch_count / total_models)
        
        return diversity