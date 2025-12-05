"""
Hugging Face Hub client with pagination, rate-limiting, and caching.
Compatible with huggingface_hub >= 0.24.0
"""
import time
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import pandas as pd
from huggingface_hub import HfApi
from tqdm.auto import tqdm


class HFMetadataClient:
    """
    Client for fetching Hugging Face model metadata with caching and rate limiting.
    """
    
    def __init__(
        self, 
        cache_db_path: Path,
        rate_limit_delay: float = 0.5,
        cache_expiry_hours: int = 24
    ):
        """
        Initialize HF client.
        
        Args:
            cache_db_path: Path to SQLite cache database
            rate_limit_delay: Seconds to wait between API calls
            cache_expiry_hours: Hours before cache is considered stale
        """
        self.api = HfApi()
        self.cache_db_path = cache_db_path
        self.rate_limit_delay = rate_limit_delay
        self.cache_expiry = timedelta(hours=cache_expiry_hours)
        
        self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_cache (
                model_id TEXT PRIMARY KEY,
                metadata TEXT NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fetched_at 
            ON model_cache(fetched_at)
        """)
        
        conn.commit()
        conn.close()
    
    def _is_cache_valid(self, fetched_at: str) -> bool:
        """Check if cached entry is still valid."""
        fetch_time = datetime.fromisoformat(fetched_at)
        return datetime.now() - fetch_time < self.cache_expiry
    
    def _get_from_cache(self, model_id: str) -> Optional[Dict]:
        """Retrieve model metadata from cache if valid."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT metadata, fetched_at FROM model_cache WHERE model_id = ?",
            (model_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result and self._is_cache_valid(result[1]):
            return json.loads(result[0])
        return None
    
    def _save_to_cache(self, model_id: str, metadata: Dict):
        """Save model metadata to cache."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO model_cache (model_id, metadata, fetched_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (model_id, json.dumps(metadata))
        )
        
        conn.commit()
        conn.close()
    
    def _extract_author_from_id(self, model_id: str) -> str:
        """
        Extract author/organization from model_id.
        Model IDs are in format: author/model-name
        """
        if '/' in model_id:
            return model_id.split('/')[0]
        return None
    
    def fetch_text_generation_models(
        self,
        limit: Optional[int] = None,
        use_cache: bool = True,
        sort_by: str = "downloads",
        languages: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch text-generation models from Hugging Face Hub.
        
        Args:
            limit: Maximum number of models to fetch (None = all)
            use_cache: Whether to use cached results
            sort_by: Sort criterion ('downloads', 'likes', 'modified')
            languages: Filter by languages (e.g., ['en', 'multilingual'])
        
        Returns:
            DataFrame with model metadata
        """
        print(f"Fetching text-generation models (limit={limit}, sort={sort_by})...")
        
        # Fetch model list - pass filters directly as kwargs
        # Note: 'library' parameter is deprecated, so we'll filter after fetching
        try:
            models = list(self.api.list_models(
                task='text-generation',
                sort=sort_by,
                direction=-1,  # Descending
                limit=limit
            ))
        except TypeError:
            # Fallback for older API versions
            models = list(self.api.list_models(
                filter='text-generation',
                sort=sort_by,
                direction=-1,
                limit=limit
            ))
        
        # Filter for transformers library
        models = [m for m in models if getattr(m, 'library_name', None) == 'transformers']
        
        print(f"Found {len(models)} models")
        
        metadata_list = []
        cache_hits = 0
        api_calls = 0
        
        for model_info in tqdm(models, desc="Fetching metadata"):
            model_id = model_info.id
            
            # Try cache first
            if use_cache:
                cached_data = self._get_from_cache(model_id)
                if cached_data:
                    metadata_list.append(cached_data)
                    cache_hits += 1
                    continue
            
            # Fetch from API
            try:
                time.sleep(self.rate_limit_delay)
                
                # Extract author from model_id (format: author/model-name)
                author = self._extract_author_from_id(model_id)
                
                # Get last_modified safely
                last_modified = None
                if hasattr(model_info, 'last_modified') and model_info.last_modified:
                    last_modified = str(model_info.last_modified)
                
                # Extract metadata
                metadata = {
                    'model_id': model_id,
                    'author': author,
                    'downloads': getattr(model_info, 'downloads', 0) or 0,
                    'likes': getattr(model_info, 'likes', 0) or 0,
                    'tags': getattr(model_info, 'tags', []) or [],
                    'pipeline_tag': getattr(model_info, 'pipeline_tag', None),
                    'library_name': getattr(model_info, 'library_name', None),
                    'created_at': str(model_info.created_at) if hasattr(model_info, 'created_at') and model_info.created_at else None,
                    'last_modified': last_modified,
                    'private': getattr(model_info, 'private', False),
                    'gated': getattr(model_info, 'gated', None),
                    'model_type': self._extract_model_type(getattr(model_info, 'tags', [])),
                    'license': self._extract_license(getattr(model_info, 'tags', [])),
                }
                
                metadata_list.append(metadata)
                
                # Cache the result
                if use_cache:
                    self._save_to_cache(model_id, metadata)
                
                api_calls += 1
                
            except Exception as e:
                print(f"\n⚠️  Error fetching {model_id}: {str(e)}")
                continue
        
        print(f"\nCache hits: {cache_hits}, API calls: {api_calls}")
        
        return pd.DataFrame(metadata_list)
    
    def _extract_model_type(self, tags: List[str]) -> Optional[str]:
        """Extract model architecture type from tags."""
        if not tags:
            return None
            
        arch_tags = [
            'llama', 'mistral', 'gpt', 'bert', 'roberta', 't5', 
            'falcon', 'mpt', 'bloom', 'opt', 'phi', 'qwen', 'gemma'
        ]
        
        for tag in tags:
            tag_lower = tag.lower()
            for arch in arch_tags:
                if arch in tag_lower:
                    return arch
        return None
    
    def _extract_license(self, tags: List[str]) -> Optional[str]:
        """Extract license from tags."""
        if not tags:
            return None
            
        license_prefixes = ['license:', 'apache', 'mit', 'gpl', 'cc-by']
        
        for tag in tags:
            tag_lower = tag.lower()
            for prefix in license_prefixes:
                if prefix in tag_lower:
                    return tag
        return None
    
    def clear_cache(self):
        """Clear all cached entries."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM model_cache")
        conn.commit()
        conn.close()
        print("✅ Cache cleared")