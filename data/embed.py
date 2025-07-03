"""
Offline Feature Extraction Script for Amazon Beauty Dataset

This script generates and caches image (ResNet-50) and text (BERT) embeddings
for the multimodal recommendation system. All embeddings are saved to data/embedded/
for efficient loading during training.

Features:
- Image embeddings using ResNet-50
- Text embeddings using BERT and Sentence-BERT
- GPU acceleration when available
- Batch processing for efficiency
- Caching to avoid recomputation
- Progress tracking and error handling
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import timm
from PIL import Image
import requests
from io import BytesIO
import pickle
import json
import logging
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional, Union
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    """Dataset for loading images from URLs."""
    
    def __init__(self, image_urls: List[str], transform=None, timeout: int = 10):
        self.image_urls = image_urls
        self.transform = transform
        self.timeout = timeout
        
    def __len__(self):
        return len(self.image_urls)
    
    def __getitem__(self, idx):
        url = self.image_urls[idx]
        try:
            # Download image
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, url, True  # True indicates success
        except Exception as e:
            # Return zero tensor for failed images
            logger.warning(f"Failed to load image {url}: {e}")
            return torch.zeros(3, 224, 224), url, False  # False indicates failure


class TextDataset(Dataset):
    """Dataset for processing text data."""
    
    def __init__(self, texts: List[str], max_length: int = 512):
        self.texts = texts
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Clean and truncate text - handle numpy arrays safely
        try:
            # Handle case where pd.isna returns array instead of scalar
            is_na = pd.isna(text)
            if hasattr(is_na, 'any'):  # If it's an array
                is_na = is_na.any()
            
            if is_na or text is None:
                text = ""
        except (ValueError, TypeError):
            # Fallback for complex objects
            if text is None or str(text).strip() == "" or str(text).lower() == "nan":
                text = ""
        
        text = str(text).strip()
        return text


class EmbeddingExtractor:
    """
    Main class for extracting image and text embeddings.
    """
    
    def __init__(self, data_dir: str = "data", batch_size: int = 32, device: str = None):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.embedded_dir = self.data_dir / "embedded"
        
        # Create embedded directory
        self.embedded_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self.batch_size = batch_size
        
        # Initialize models (lazy loading)
        self.resnet_model = None
        self.bert_model = None
        self.sentence_bert_model = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load processed data."""
        logger.info("Loading processed data")
        
        # Load interactions (for text)
        interactions_df = pd.read_parquet(self.processed_dir / "interactions_full.parquet")
        
        # Load metadata (for product text)
        metadata_df = pd.read_parquet(self.processed_dir / "metadata.parquet")
        
        # Load images
        images_df = pd.read_parquet(self.processed_dir / "images.parquet")
        
        logger.info(f"Loaded {len(interactions_df)} interactions")
        logger.info(f"Loaded {len(metadata_df)} metadata records")
        logger.info(f"Loaded {len(images_df)} image URLs")
        
        return interactions_df, metadata_df, images_df
    
    def load_resnet_model(self) -> nn.Module:
        """Load ResNet-50 model for image embeddings."""
        if self.resnet_model is not None:
            return self.resnet_model
        
        logger.info("Loading ResNet-50 model")
        
        # Load pretrained ResNet-50 without final classification layer
        self.resnet_model = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.resnet_model.eval()
        self.resnet_model.to(self.device)
        
        # Define image preprocessing transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("ResNet-50 model loaded successfully")
        return self.resnet_model
    
    def load_bert_model(self):
        """Load BERT model for text embeddings."""
        if self.bert_model is not None:
            return self.bert_model, self.bert_tokenizer
        
        logger.info("Loading BERT model")
        
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        self.bert_model.to(self.device)
        
        logger.info("BERT model loaded successfully")
        return self.bert_model, self.bert_tokenizer
    
    def load_sentence_bert_model(self):
        """Load Sentence-BERT model for text embeddings."""
        if self.sentence_bert_model is not None:
            return self.sentence_bert_model
        
        logger.info("Loading Sentence-BERT model")
        
        self.sentence_bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_bert_model.to(self.device)
        
        logger.info("Sentence-BERT model loaded successfully")
        return self.sentence_bert_model
    
    def extract_image_embeddings(self, images_df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Extract ResNet-50 image embeddings for all items."""
        logger.info("Extracting ResNet-50 image embeddings")
        
        # Check cache
        cache_file = self.embedded_dir / "image_embeddings_resnet50.pkl"
        if cache_file.exists():
            logger.info(f"Loading cached image embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        model = self.load_resnet_model()
        embeddings = {}
        
        # Group images by item for averaging
        item_groups = images_df.groupby('item_idx')
        
        for item_idx, group in tqdm(item_groups, desc="Processing items"):
            # Prioritize high resolution images
            hi_res_urls = group[group['resolution'] == 'hi_res']['image_url'].tolist()
            large_urls = group[group['resolution'] == 'large']['image_url'].tolist()
            thumb_urls = group[group['resolution'] == 'thumb']['image_url'].tolist()
            
            # Select best images (prioritize hi-res, limit to 3 per item)
            selected_urls = (hi_res_urls[:2] + large_urls[:1] + thumb_urls[:1])[:3]
            
            if not selected_urls:
                continue
            
            # Create dataset and dataloader
            dataset = ImageDataset(selected_urls, transform=self.image_transform)
            dataloader = DataLoader(dataset, batch_size=min(len(selected_urls), 4), shuffle=False)
            
            item_embeddings = []
            
            with torch.no_grad():
                for batch_images, batch_urls, batch_success in dataloader:
                    batch_images = batch_images.to(self.device)
                    
                    # Filter successful images
                    valid_mask = batch_success
                    if not valid_mask.any():
                        continue
                    
                    valid_images = batch_images[valid_mask]
                    
                    # Extract features using ResNet-50
                    features = model(valid_images)
                    batch_embeddings = features.cpu().numpy()
                    
                    item_embeddings.append(batch_embeddings)
            
            if item_embeddings:
                # Average embeddings across all images for this item
                all_embeddings = np.vstack(item_embeddings)
                avg_embedding = np.mean(all_embeddings, axis=0)
                embeddings[item_idx] = avg_embedding
            else:
                logger.warning(f"No valid images found for item {item_idx}")
        
        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        logger.info(f"Extracted ResNet-50 embeddings for {len(embeddings)} items")
        return embeddings
    
    def extract_bert_embeddings(self, texts: List[str], text_type: str = "review") -> np.ndarray:
        """Extract BERT embeddings for texts."""
        logger.info(f"Extracting BERT embeddings for {text_type}")
        
        # Check cache
        text_hash = hashlib.md5(str(texts[:100]).encode()).hexdigest()[:8]  # Hash first 100 texts
        cache_file = self.embedded_dir / f"bert_embeddings_{text_type}_{text_hash}.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached BERT embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        model, tokenizer = self.load_bert_model()
        embeddings = []
        
        # Process in batches
        dataset = TextDataset(texts)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_texts in tqdm(dataloader, desc=f"Processing {text_type}"):
                # Tokenize
                encoded = tokenizer(
                    batch_texts, truncation=True, padding=True, 
                    max_length=512, return_tensors='pt'
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Get BERT outputs
                outputs = model(**encoded)
                
                # Use CLS token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(cls_embeddings)
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        logger.info(f"Extracted BERT embeddings: {embeddings.shape}")
        return embeddings
    
    def extract_sentence_bert_embeddings(self, texts: List[str], text_type: str = "review") -> np.ndarray:
        """Extract Sentence-BERT embeddings for texts."""
        logger.info(f"Extracting Sentence-BERT embeddings for {text_type}")
        
        # Check cache
        text_hash = hashlib.md5(str(texts[:100]).encode()).hexdigest()[:8]
        cache_file = self.embedded_dir / f"sentence_bert_embeddings_{text_type}_{text_hash}.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached Sentence-BERT embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        model = self.load_sentence_bert_model()
        
        # Clean and ensure texts are proper strings
        cleaned_texts = []
        for text in texts:
            try:
                # Handle case where pd.isna returns array instead of scalar
                is_na = pd.isna(text)
                if hasattr(is_na, 'any'):  # If it's an array
                    is_na = is_na.any()
                
                if is_na or text is None:
                    cleaned_texts.append("")
                else:
                    # Convert to string and clean
                    clean_text = str(text).strip()
                    cleaned_texts.append(clean_text)
            except (ValueError, TypeError):
                # Fallback for complex objects
                if text is None or str(text).strip() == "" or str(text).lower() == "nan":
                    cleaned_texts.append("")
                else:
                    cleaned_texts.append(str(text).strip())
        
        # Process in batches
        embeddings = []
        batch_size = self.batch_size
        
        for i in tqdm(range(0, len(cleaned_texts), batch_size), desc=f"Processing {text_type}"):
            batch_texts = cleaned_texts[i:i + batch_size]
            
            # Ensure batch_texts is a list of strings
            batch_texts = [str(text) for text in batch_texts]
            
            batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        
        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        logger.info(f"Extracted Sentence-BERT embeddings: {embeddings.shape}")
        return embeddings
    
    def prepare_review_embeddings(self, interactions_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract review text embeddings."""
        logger.info("Preparing review embeddings")
        
        # Extract review texts
        review_texts = interactions_df['text'].fillna('').tolist()
        
        review_embeddings = {}
        
        # Check for cached BERT embeddings
        bert_hash = hashlib.md5(str(review_texts[:100]).encode()).hexdigest()[:8]
        bert_cache = self.embedded_dir / f"bert_embeddings_review_{bert_hash}.pkl"
        
        if bert_cache.exists():
            logger.info("✅ Review BERT embeddings already cached, skipping")
            with open(bert_cache, 'rb') as f:
                review_embeddings['bert'] = pickle.load(f)
        else:
            logger.info("🔄 Extracting review BERT embeddings")
            review_embeddings['bert'] = self.extract_bert_embeddings(review_texts, "review")
        
        # Check for cached Sentence-BERT embeddings
        sbert_hash = hashlib.md5(str(review_texts[:100]).encode()).hexdigest()[:8]
        sbert_cache = self.embedded_dir / f"sentence_bert_embeddings_review_{sbert_hash}.pkl"
        
        if sbert_cache.exists():
            logger.info("✅ Review Sentence-BERT embeddings already cached, skipping")
            with open(sbert_cache, 'rb') as f:
                review_embeddings['sentence_bert'] = pickle.load(f)
        else:
            logger.info("🔄 Extracting review Sentence-BERT embeddings")
            review_embeddings['sentence_bert'] = self.extract_sentence_bert_embeddings(review_texts, "review")
        
        return review_embeddings
    
    def prepare_item_text_embeddings(self, metadata_df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract item text embeddings from metadata."""
        logger.info("Preparing item text embeddings")
        
        item_embeddings = {}
        
        # Prepare different text fields
        text_fields = {
            'title': metadata_df['title'].fillna('').tolist(),
            'description': metadata_df['description'].fillna('').tolist(),
            'combined': (metadata_df['title'].fillna('') + ' ' + metadata_df['description'].fillna('')).tolist()
        }
        
        for field_name, texts in text_fields.items():
            logger.info(f"Processing field: {field_name}")
            
            # Check what embeddings are already cached for this field
            item_embeddings[field_name] = {}
            
            # BERT embeddings
            bert_hash = hashlib.md5(str(texts[:100]).encode()).hexdigest()[:8]
            bert_cache = self.embedded_dir / f"bert_embeddings_item_{field_name}_{bert_hash}.pkl"
            
            if bert_cache.exists():
                logger.info(f"✅ BERT embeddings for {field_name} already cached, skipping")
                with open(bert_cache, 'rb') as f:
                    item_embeddings[field_name]['bert'] = pickle.load(f)
            else:
                logger.info(f"🔄 Extracting BERT embeddings for {field_name}")
                item_embeddings[field_name]['bert'] = self.extract_bert_embeddings(texts, f"item_{field_name}")
            
            # Sentence-BERT embeddings  
            sbert_hash = hashlib.md5(str(texts[:100]).encode()).hexdigest()[:8]
            sbert_cache = self.embedded_dir / f"sentence_bert_embeddings_item_{field_name}_{sbert_hash}.pkl"
            
            if sbert_cache.exists():
                logger.info(f"✅ Sentence-BERT embeddings for {field_name} already cached, skipping")
                with open(sbert_cache, 'rb') as f:
                    item_embeddings[field_name]['sentence_bert'] = pickle.load(f)
            else:
                logger.info(f"🔄 Extracting Sentence-BERT embeddings for {field_name}")
                item_embeddings[field_name]['sentence_bert'] = self.extract_sentence_bert_embeddings(texts, f"item_{field_name}")
        
        return item_embeddings
    
    def save_embedding_matrices(self, review_embeddings: Dict[str, np.ndarray], 
                              item_embeddings: Dict[str, Dict[str, np.ndarray]],
                              image_embeddings: Dict[int, np.ndarray],
                              interactions_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
        """Save embedding matrices for efficient loading."""
        logger.info("Saving embedding matrices")
        
        # Get dimensions
        n_users = interactions_df['user_idx'].nunique()
        n_items = metadata_df['item_idx'].nunique()
        
        # Save review embeddings
        review_file = self.embedded_dir / "review_embeddings.npz"
        review_data = {
            'interaction_indices': interactions_df.index.values,
            'user_indices': interactions_df['user_idx'].values,
            'item_indices': interactions_df['item_idx'].values
        }
        
        for model_name, embeddings in review_embeddings.items():
            review_data[f'embeddings_{model_name}'] = embeddings
        
        np.savez_compressed(review_file, **review_data)
        logger.info(f"Saved review embeddings to {review_file}")
        
        # Save item text embeddings
        for field_name, field_embeddings in item_embeddings.items():
            for model_name, embeddings in field_embeddings.items():
                # Create item-indexed matrix
                embedding_matrix = np.zeros((n_items, embeddings.shape[1]))
                
                for idx, embedding in enumerate(embeddings):
                    item_idx = metadata_df.iloc[idx]['item_idx']
                    embedding_matrix[item_idx] = embedding
                
                item_file = self.embedded_dir / f"item_text_{field_name}_{model_name}.npy"
                np.save(item_file, embedding_matrix)
                logger.info(f"Saved {field_name} {model_name} embeddings to {item_file}")
        
        # Save image embeddings
        if image_embeddings:
            embedding_dim = list(image_embeddings.values())[0].shape[0]
            image_matrix = np.zeros((n_items, embedding_dim))
            
            for item_idx, embedding in image_embeddings.items():
                image_matrix[item_idx] = embedding
            
            image_file = self.embedded_dir / "image_embeddings_resnet50.npy"
            np.save(image_file, image_matrix)
            logger.info(f"Saved ResNet-50 image embeddings to {image_file}")
    
    def generate_embedding_summary(self, review_embeddings: Dict[str, np.ndarray],
                                 item_embeddings: Dict[str, Dict[str, np.ndarray]],
                                 image_embeddings: Dict[int, np.ndarray]) -> None:
        """Generate summary statistics."""
        logger.info("Generating embedding summary")
        
        summary = {
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'device_used': str(self.device),
            'review_embeddings': {},
            'item_text_embeddings': {},
            'image_embeddings': {}
        }
        
        # Review embeddings summary
        for model_name, embeddings in review_embeddings.items():
            summary['review_embeddings'][model_name] = {
                'shape': list(embeddings.shape),
                'dtype': str(embeddings.dtype),
                'mean_norm': float(np.linalg.norm(embeddings, axis=1).mean())
            }
        
        # Item text embeddings summary
        for field_name, field_embeddings in item_embeddings.items():
            summary['item_text_embeddings'][field_name] = {}
            for model_name, embeddings in field_embeddings.items():
                summary['item_text_embeddings'][field_name][model_name] = {
                    'shape': list(embeddings.shape),
                    'dtype': str(embeddings.dtype),
                    'mean_norm': float(np.linalg.norm(embeddings, axis=1).mean())
                }
        
        # Image embeddings summary
        if image_embeddings:
            embeddings_array = np.array(list(image_embeddings.values()))
            summary['image_embeddings'] = {
                'model': 'resnet50',
                'n_items': len(image_embeddings),
                'embedding_dim': embeddings_array.shape[1],
                'mean_norm': float(np.linalg.norm(embeddings_array, axis=1).mean())
            }
        
        # Save summary
        summary_file = self.embedded_dir / "embedding_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Embedding summary saved to {summary_file}")
        
        # Log key statistics
        logger.info("=== EMBEDDING SUMMARY ===")
        for model_name, stats in summary['review_embeddings'].items():
            logger.info(f"Review embeddings ({model_name}): {stats['shape']}")
        
        for field_name, field_stats in summary['item_text_embeddings'].items():
            for model_name, stats in field_stats.items():
                logger.info(f"Item {field_name} ({model_name}): {stats['shape']}")
        
        if summary['image_embeddings']:
            stats = summary['image_embeddings']
            logger.info(f"Image embeddings (ResNet-50): {stats['n_items']} items, dim {stats['embedding_dim']}")
    
    def run_extraction(self, extract_images: bool = True, extract_text: bool = True) -> None:
        """
        Run the complete embedding extraction pipeline.
        
        Args:
            extract_images: Whether to extract ResNet-50 image embeddings
            extract_text: Whether to extract BERT/Sentence-BERT text embeddings
        """
        logger.info("Starting embedding extraction pipeline")
        logger.info(f"Image extraction: {extract_images}")
        logger.info(f"Text extraction: {extract_text}")
        
        try:
            # 1. Load data
            interactions_df, metadata_df, images_df = self.load_data()
            
            # 2. Extract embeddings
            review_embeddings = {}
            item_embeddings = {}
            image_embeddings = {}
            
            if extract_text:
                # Extract review embeddings
                review_embeddings = self.prepare_review_embeddings(interactions_df)
                
                # Extract item text embeddings
                item_embeddings = self.prepare_item_text_embeddings(metadata_df)
            
            if extract_images:
                # Extract ResNet-50 image embeddings
                image_embeddings = self.extract_image_embeddings(images_df)
            
            # 3. Save embeddings
            self.save_embedding_matrices(review_embeddings, item_embeddings, image_embeddings,
                                       interactions_df, metadata_df)
            
            # 4. Generate summary
            self.generate_embedding_summary(review_embeddings, item_embeddings, image_embeddings)
            
            logger.info("Embedding extraction completed successfully!")
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            raise


def main():
    """
    Main function to run embedding extraction.
    """
    # Configuration
    config = {
        'batch_size': 16,  # Adjust based on GPU memory
        'extract_images': True,  # ResNet-50 image embeddings
        'extract_text': True     # BERT + Sentence-BERT text embeddings
    }
    
    # Create extractor
    extractor = EmbeddingExtractor(batch_size=config['batch_size'])
    
    # Run extraction
    extractor.run_extraction(
        extract_images=config['extract_images'],
        extract_text=config['extract_text']
    )


if __name__ == "__main__":
    main()
