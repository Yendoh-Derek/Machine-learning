"""
Enhanced model card parser for extracting information from HuggingFace README files.
Includes parameter extraction, quantization detection, and improved text cleaning.
"""
import re
import yaml
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from huggingface_hub import HfApi
from tqdm.auto import tqdm


@dataclass
class ModelCard:
    """Structured representation of a model card."""
    model_id: str
    raw_text: str
    yaml_metadata: Dict
    description: Optional[str]
    use_cases: Optional[str]
    limitations: Optional[str]
    training_details: Optional[str]
    card_length: int
    has_yaml: bool
    # New fields for cost proxies
    num_parameters: Optional[float]  # In billions
    quantization: Optional[str]  # e.g., "4-bit", "8-bit", "FP16", None
    model_size_category: Optional[str]  # "small", "medium", "large", "xlarge"


class ModelCardFetcher:
    """
    Fetches and parses model cards from HuggingFace Hub.
    """
    
    def __init__(
        self,
        cache_dir: Path,
        rate_limit_delay: float = 0.3
    ):
        """
        Initialize card fetcher.
        
        Args:
            cache_dir: Directory to cache model card text files
            rate_limit_delay: Seconds to wait between API calls
        """
        self.api = HfApi()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
    
    def _get_cache_path(self, model_id: str) -> Path:
        """Get cache file path for a model."""
        # Replace / with -- for filesystem safety
        safe_id = model_id.replace('/', '--')
        return self.cache_dir / f"{safe_id}.md"
    
    def _load_from_cache(self, model_id: str) -> Optional[str]:
        """Load model card from cache if exists."""
        cache_path = self._get_cache_path(model_id)
        if cache_path.exists():
            return cache_path.read_text(encoding='utf-8')
        return None
    
    def _save_to_cache(self, model_id: str, content: str):
        """Save model card to cache."""
        cache_path = self._get_cache_path(model_id)
        cache_path.write_text(content, encoding='utf-8')
    
    def fetch_model_card(
        self,
        model_id: str,
        use_cache: bool = True
    ) -> Optional[str]:
        """
        Fetch model card (README.md) for a single model.
        
        Args:
            model_id: HuggingFace model ID
            use_cache: Whether to use cached version
            
        Returns:
            Model card text or None if not available
        """
        # Try cache first
        if use_cache:
            cached = self._load_from_cache(model_id)
            if cached:
                return cached
        
        # Fetch from API
        try:
            time.sleep(self.rate_limit_delay)
            
            # Get model card using HF API
            card_data = self.api.model_info(model_id, files_metadata=False)
            
            if hasattr(card_data, 'card_data') and card_data.card_data:
                # Try to get the actual README content
                try:
                    from huggingface_hub import hf_hub_download
                    readme_path = hf_hub_download(
                        repo_id=model_id,
                        filename="README.md",
                        repo_type="model"
                    )
                    content = Path(readme_path).read_text(encoding='utf-8')
                except:
                    # Fallback: use card_data as text
                    content = str(card_data.card_data)
            else:
                return None
            
            # Cache the result
            if use_cache and content:
                self._save_to_cache(model_id, content)
            
            return content
            
        except Exception as e:
            return None
    
    def fetch_batch(
        self,
        model_ids: List[str],
        use_cache: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Optional[str]]:
        """
        Fetch model cards for multiple models.
        
        Args:
            model_ids: List of model IDs
            use_cache: Whether to use cache
            show_progress: Show progress bar
            
        Returns:
            Dictionary mapping model_id to card text
        """
        results = {}
        iterator = tqdm(model_ids, desc="Fetching model cards") if show_progress else model_ids
        
        for model_id in iterator:
            results[model_id] = self.fetch_model_card(model_id, use_cache)
        
        return results


class ModelCardParser:
    """
    Enhanced parser for model cards with improved text cleaning and metadata extraction.
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean HTML tags, markdown links, and badges from text.
        
        Args:
            text: Raw text with potential HTML/markdown
            
        Returns:
            Cleaned plain text
        """
        if not text:
            return text
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove markdown image syntax ![alt](url)
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
        
        # Remove markdown links but keep text [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove badge URLs (common pattern)
        text = re.sub(r'https?://img\.shields\.io[^\s]+', '', text)
        
        # Remove other URLs
        text = re.sub(r'https?://[^\s]+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def parse_yaml_frontmatter(card_text: str) -> Dict:
        """
        Extract YAML frontmatter from model card.
        
        Args:
            card_text: Full model card text
            
        Returns:
            Dictionary of YAML metadata
        """
        if not card_text:
            return {}
        
        # Look for YAML frontmatter between --- markers
        yaml_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(yaml_pattern, card_text, re.DOTALL)
        
        if match:
            yaml_text = match.group(1)
            try:
                return yaml.safe_load(yaml_text) or {}
            except yaml.YAMLError:
                return {}
        
        return {}
    
    @staticmethod
    def extract_section(card_text: str, section_names: List[str]) -> Optional[str]:
        """
        Extract a specific section from model card with multiple possible names.
        
        Args:
            card_text: Full model card text
            section_names: List of possible section header names (case-insensitive)
            
        Returns:
            Section content or None
        """
        if not card_text:
            return None
        
        # Remove YAML frontmatter first
        card_text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', card_text, flags=re.DOTALL)
        
        # Try each section name
        for section_name in section_names:
            # Pattern to match section headers (## or ### or #)
            pattern = rf'#+\s*{re.escape(section_name)}\s*\n(.*?)(?=\n#+\s|\Z)'
            match = re.search(pattern, card_text, re.IGNORECASE | re.DOTALL)
            
            if match:
                content = match.group(1).strip()
                # Clean HTML/markdown
                content = ModelCardParser.clean_text(content)
                # Limit to first 500 chars for manageability
                return content[:500] if len(content) > 500 else content
        
        return None
    
    @staticmethod
    def extract_description(card_text: str) -> Optional[str]:
        """
        Extract model description with improved cleaning.
        
        Args:
            card_text: Full model card text
            
        Returns:
            Cleaned description text or None
        """
        if not card_text:
            return None
        
        # Remove YAML frontmatter
        card_text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', card_text, flags=re.DOTALL)
        
        # Remove title (first # header)
        card_text = re.sub(r'^#\s+.*?\n', '', card_text, flags=re.MULTILINE)
        
        # Get text before first ## header
        paragraphs = re.split(r'\n##', card_text, maxsplit=1)[0].strip()
        
        # Clean HTML/markdown
        paragraphs = ModelCardParser.clean_text(paragraphs)
        
        # Get first substantial paragraph
        for para in paragraphs.split('\n\n'):
            para = para.strip()
            # At least 50 chars and not just badges/links
            if para and len(para) > 50 and not para.startswith('http'):
                # Limit length
                return para[:500] if len(para) > 500 else para
        
        # Fallback: return first 500 chars if nothing found
        if paragraphs and len(paragraphs) > 50:
            return paragraphs[:500]
        
        return None
    
    @staticmethod
    def extract_parameter_count(card_text: str, model_id: str) -> Optional[float]:
        """
        Extract number of parameters from model card or model ID.
        
        Args:
            card_text: Full model card text
            model_id: Model identifier
            
        Returns:
            Parameter count in billions, or None
        """
        if not card_text and not model_id:
            return None
        
        # Common patterns for parameter counts
        patterns = [
            r'(\d+\.?\d*)\s*[Bb]illion?\s+parameters?',  # "7 billion parameters"
            r'(\d+\.?\d*)\s*B\s+parameters?',  # "7B parameters"
            r'(\d+\.?\d*)\s*[Bb]illions?\s+params?',
            r'(\d+)\s*[Bb]\b',  # "7B" standalone
            r'(\d+\.?\d*)\s*[Mm]illion?\s+parameters?',  # "600 million parameters"
            r'(\d+\.?\d*)\s*M\s+parameters?',  # "600M parameters"
        ]
        
        # Try model ID first (e.g., "Qwen2.5-7B-Instruct" -> 7B)
        id_match = re.search(r'[-_](\d+\.?\d*)[Bb]\b', model_id)
        if id_match:
            return float(id_match.group(1))
        
        # Try model card
        if card_text:
            for pattern in patterns:
                match = re.search(pattern, card_text)
                if match:
                    value = float(match.group(1))
                    # Convert millions to billions
                    if 'million' in pattern.lower() or 'M' in pattern:
                        value = value / 1000
                    return value
        
        return None
    
    @staticmethod
    def detect_quantization(card_text: str, model_id: str, tags: Optional[List[str]]) -> Optional[str]:
        """
        Detect quantization level from model card, ID, or tags.
        
        Args:
            card_text: Full model card text
            model_id: Model identifier
            tags: Model tags (can be None, list, or array)
            
        Returns:
            Quantization type (e.g., "4-bit", "8-bit", "FP16") or None
        """
        # Safely handle tags
        if tags is None:
            tags = []
        elif not isinstance(tags, list):
            # Convert numpy array or other iterable to list
            try:
                tags = list(tags)
            except:
                tags = []
        
        # Check model ID for quantization indicators
        id_lower = model_id.lower()
        
        quant_patterns = {
            'GPTQ': r'gptq',
            'AWQ': r'awq',
            '4-bit': r'4bit|q4',
            '8-bit': r'8bit|q8',
            '2-bit': r'2bit|q2',
            'FP16': r'fp16',
            'FP8': r'fp8',
            'INT8': r'int8',
            'INT4': r'int4',
        }
        
        # Check model ID
        for quant_type, pattern in quant_patterns.items():
            if re.search(pattern, id_lower):
                return quant_type
        
        # Check tags
        if tags and len(tags) > 0:
            for tag in tags:
                if tag:  # Skip None/empty tags
                    tag_lower = str(tag).lower()
                    for quant_type, pattern in quant_patterns.items():
                        if re.search(pattern, tag_lower):
                            return quant_type
        
        # Check card text
        if card_text:
            card_lower = card_text.lower()
            for quant_type, pattern in quant_patterns.items():
                if re.search(pattern, card_lower):
                    return quant_type
        
        return None
    
    @staticmethod
    def categorize_model_size(num_parameters: Optional[float]) -> Optional[str]:
        """
        Categorize model by parameter count.
        
        Args:
            num_parameters: Parameter count in billions
            
        Returns:
            Size category: "tiny", "small", "medium", "large", "xlarge"
        """
        if num_parameters is None:
            return None
        
        if num_parameters < 1:
            return "tiny"  # < 1B
        elif num_parameters < 3:
            return "small"  # 1-3B
        elif num_parameters < 10:
            return "medium"  # 3-10B
        elif num_parameters < 30:
            return "large"  # 10-30B
        else:
            return "xlarge"  # 30B+
    
    @classmethod
    def parse_full_card(
        cls, 
        model_id: str, 
        card_text: str,
        tags: Optional[List[str]] = None
    ) -> ModelCard:
        """
        Parse full model card into structured format with enhanced metadata.
        
        Args:
            model_id: Model identifier
            card_text: Raw card text
            tags: Model tags (for quantization detection) - can be None, list, or array
            
        Returns:
            ModelCard object with extracted fields
        """
        # Safely handle tags early
        if tags is None:
            tags = []
        elif not isinstance(tags, list):
            try:
                tags = list(tags)
            except:
                tags = []
        
        # Safely handle card_text
        if card_text is None:
            card_text = ""
        elif not isinstance(card_text, str):
            try:
                card_text = str(card_text)
            except:
                card_text = ""
        
        if not card_text or len(card_text) == 0:
            return ModelCard(
                model_id=model_id,
                raw_text="",
                yaml_metadata={},
                description=None,
                use_cases=None,
                limitations=None,
                training_details=None,
                card_length=0,
                has_yaml=False,
                num_parameters=None,
                quantization=None,
                model_size_category=None
            )
        
        yaml_metadata = cls.parse_yaml_frontmatter(card_text)
        
        # Extract parameter count
        num_params = cls.extract_parameter_count(card_text, model_id)
        
        # Detect quantization
        quantization = cls.detect_quantization(card_text, model_id, tags)
        
        # Categorize size
        size_category = cls.categorize_model_size(num_params)
        
        return ModelCard(
            model_id=model_id,
            raw_text=card_text,
            yaml_metadata=yaml_metadata,
            description=cls.extract_description(card_text),
            use_cases=cls.extract_section(card_text, ["uses", "use cases", "usage", "applications"]),
            limitations=cls.extract_section(card_text, ["limitations", "limits", "bias", "risks"]),
            training_details=cls.extract_section(card_text, ["training", "training data", "training procedure", "training details"]),
            card_length=len(card_text),
            has_yaml=len(yaml_metadata) > 0,
            num_parameters=num_params,
            quantization=quantization,
            model_size_category=size_category
        )