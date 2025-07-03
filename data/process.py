"""
Data Processing Script for Amazon Beauty Dataset

This script processes the raw ingested data to create clean train/val/test splits
for the multimodal recommendation system.

Key steps:
1. Clean missing IDs and filter data
2. Handle cold-start users/items (minimum interaction threshold)
3. Build user-item interaction matrices
4. Create chronological splits
5. Save processed data to data/processed/
"""

import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonBeautyProcessor:
    """
    Handles processing of Amazon Beauty dataset for recommendation system.
    """
    
    def __init__(self, data_dir: str = "data", min_interactions: int = 5):
        self.data_dir = Path(data_dir)
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        
        # Create processed directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters
        self.min_interactions = min_interactions
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        
        # Encoders for consistent ID mapping
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Data storage
        self.reviews_df = None
        self.metadata_df = None
        self.images_df = None
        
    def load_data(self) -> None:
        """Load data from interim directory."""
        logger.info("Loading data from interim directory")
        
        # Load reviews
        reviews_path = self.interim_dir / "reviews.parquet"
        self.reviews_df = pd.read_parquet(reviews_path)
        logger.info(f"Loaded {len(self.reviews_df)} reviews")
        
        # Load metadata
        metadata_path = self.interim_dir / "metadata.parquet"
        self.metadata_df = pd.read_parquet(metadata_path)
        logger.info(f"Loaded {len(self.metadata_df)} metadata records")
        
        # Load images
        images_path = self.interim_dir / "images.parquet"
        self.images_df = pd.read_parquet(images_path)
        logger.info(f"Loaded {len(self.images_df)} image URLs")
        
    def clean_and_filter_data(self) -> pd.DataFrame:
        """
        Clean data and filter for cold-start handling.
        
        Returns:
            pd.DataFrame: Cleaned interactions dataframe
        """
        logger.info("Cleaning and filtering data")
        
        # Start with reviews data
        interactions_df = self.reviews_df.copy()
        
        # Remove any rows with missing critical IDs (should be none based on analysis)
        initial_count = len(interactions_df)
        interactions_df = interactions_df.dropna(subset=['user_id', 'parent_asin', 'rating', 'timestamp'])
        logger.info(f"Removed {initial_count - len(interactions_df)} rows with missing critical data")
        
        # Convert timestamp to datetime for easier handling
        interactions_df['datetime'] = pd.to_datetime(interactions_df['timestamp'], unit='ms')
        
        # Filter for minimum interactions per user and item
        logger.info(f"Filtering for minimum {self.min_interactions} interactions per user/item")
        
        # Iteratively filter until convergence
        prev_users, prev_items = 0, 0
        iteration = 0
        
        while True:
            iteration += 1
            
            # Count interactions per user and item
            user_counts = interactions_df['user_id'].value_counts()
            item_counts = interactions_df['parent_asin'].value_counts()
            
            # Find users and items with sufficient interactions
            valid_users = user_counts[user_counts >= self.min_interactions].index
            valid_items = item_counts[item_counts >= self.min_interactions].index
            
            # Filter interactions
            interactions_df = interactions_df[
                (interactions_df['user_id'].isin(valid_users)) & 
                (interactions_df['parent_asin'].isin(valid_items))
            ]
            
            curr_users = len(valid_users)
            curr_items = len(valid_items)
            
            logger.info(f"Iteration {iteration}: {curr_users} users, {curr_items} items, {len(interactions_df)} interactions")
            
            # Check for convergence
            if curr_users == prev_users and curr_items == prev_items:
                break
                
            prev_users, prev_items = curr_users, curr_items
            
            # Safety check to prevent infinite loops
            if iteration > 10:
                logger.warning("Maximum iterations reached in filtering")
                break
        
        # Log final statistics
        logger.info(f"Final dataset: {len(interactions_df)} interactions")
        logger.info(f"Users: {interactions_df['user_id'].nunique()}")
        logger.info(f"Items: {interactions_df['parent_asin'].nunique()}")
        
        # Sort by timestamp for chronological processing
        interactions_df = interactions_df.sort_values('timestamp')
        
        return interactions_df
    
    def create_user_item_encodings(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create consistent user and item ID encodings.
        
        Args:
            interactions_df: Cleaned interactions dataframe
            
        Returns:
            pd.DataFrame: Interactions with encoded IDs
        """
        logger.info("Creating user and item encodings")
        
        # Fit encoders
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['parent_asin'].unique()
        
        self.user_encoder.fit(unique_users)
        self.item_encoder.fit(unique_items)
        
        # Apply encodings
        interactions_df['user_idx'] = self.user_encoder.transform(interactions_df['user_id'])
        interactions_df['item_idx'] = self.item_encoder.transform(interactions_df['parent_asin'])
        
        logger.info(f"Encoded {len(unique_users)} users and {len(unique_items)} items")
        
        return interactions_df
    
    def create_chronological_splits(self, interactions_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create chronological train/val/test splits.
        
        Args:
            interactions_df: Interactions with encoded IDs
            
        Returns:
            Dict containing train, val, test dataframes
        """
        logger.info("Creating chronological splits")
        
        # Sort by timestamp
        interactions_df = interactions_df.sort_values('timestamp')
        
        # Calculate split points
        n_interactions = len(interactions_df)
        train_end = int(n_interactions * self.train_ratio)
        val_end = int(n_interactions * (self.train_ratio + self.val_ratio))
        
        # Create splits
        train_df = interactions_df.iloc[:train_end].copy()
        val_df = interactions_df.iloc[train_end:val_end].copy()
        test_df = interactions_df.iloc[val_end:].copy()
        
        # Log split information
        logger.info(f"Train: {len(train_df)} interactions ({len(train_df)/n_interactions*100:.1f}%)")
        logger.info(f"Val: {len(val_df)} interactions ({len(val_df)/n_interactions*100:.1f}%)")
        logger.info(f"Test: {len(test_df)} interactions ({len(test_df)/n_interactions*100:.1f}%)")
        
        # Log time ranges
        logger.info(f"Train time range: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
        logger.info(f"Val time range: {val_df['datetime'].min()} to {val_df['datetime'].max()}")
        logger.info(f"Test time range: {test_df['datetime'].min()} to {test_df['datetime'].max()}")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def build_interaction_matrices(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, sparse.csr_matrix]:
        """
        Build sparse interaction matrices for each split.
        
        Args:
            splits: Dictionary of train/val/test dataframes
            
        Returns:
            Dict of sparse matrices
        """
        logger.info("Building interaction matrices")
        
        # Get dimensions
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        matrices = {}
        
        for split_name, df in splits.items():
            # Create sparse matrix
            matrix = sparse.csr_matrix(
                (df['rating'].values, (df['user_idx'].values, df['item_idx'].values)),
                shape=(n_users, n_items)
            )
            matrices[split_name] = matrix
            
            logger.info(f"{split_name} matrix shape: {matrix.shape}, density: {matrix.nnz / (n_users * n_items):.6f}")
        
        return matrices
    
    def prepare_metadata(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare metadata for items in the filtered dataset.
        
        Args:
            interactions_df: Filtered interactions dataframe
            
        Returns:
            pd.DataFrame: Filtered metadata
        """
        logger.info("Preparing metadata")
        
        # Get items in the filtered dataset
        valid_items = interactions_df['parent_asin'].unique()
        
        # Filter metadata
        filtered_metadata = self.metadata_df[
            self.metadata_df['parent_asin'].isin(valid_items)
        ].copy()
        
        # Add item indices
        filtered_metadata['item_idx'] = self.item_encoder.transform(filtered_metadata['parent_asin'])
        
        logger.info(f"Metadata for {len(filtered_metadata)} items")
        
        return filtered_metadata
    
    def prepare_images(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare image data for items in the filtered dataset.
        
        Args:
            interactions_df: Filtered interactions dataframe
            
        Returns:
            pd.DataFrame: Filtered image data
        """
        logger.info("Preparing image data")
        
        # Get items in the filtered dataset
        valid_items = interactions_df['parent_asin'].unique()
        
        # Filter images
        filtered_images = self.images_df[
            self.images_df['parent_asin'].isin(valid_items)
        ].copy()
        
        # Add item indices
        filtered_images['item_idx'] = self.item_encoder.transform(filtered_images['parent_asin'])
        
        logger.info(f"Images for {len(filtered_images)} records")
        
        return filtered_images
    
    def save_processed_data(self, interactions_df: pd.DataFrame, splits: Dict[str, pd.DataFrame], 
                          matrices: Dict[str, sparse.csr_matrix], metadata_df: pd.DataFrame,
                          images_df: pd.DataFrame) -> None:
        """
        Save all processed data to the processed directory.
        
        Args:
            interactions_df: Full filtered interactions
            splits: Train/val/test splits
            matrices: Sparse interaction matrices
            metadata_df: Filtered metadata
            images_df: Filtered image data
        """
        logger.info("Saving processed data")
        
        # Save interaction splits
        for split_name, df in splits.items():
            output_path = self.processed_dir / f"interactions_{split_name}.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved {split_name} interactions: {len(df)} records")
        
        # Save full interactions
        full_path = self.processed_dir / "interactions_full.parquet"
        interactions_df.to_parquet(full_path, index=False)
        logger.info(f"Saved full interactions: {len(interactions_df)} records")
        
        # Save sparse matrices
        for split_name, matrix in matrices.items():
            matrix_path = self.processed_dir / f"matrix_{split_name}.npz"
            sparse.save_npz(matrix_path, matrix)
            logger.info(f"Saved {split_name} matrix: {matrix.shape}")
        
        # Save metadata and images
        metadata_path = self.processed_dir / "metadata.parquet"
        metadata_df.to_parquet(metadata_path, index=False)
        logger.info(f"Saved metadata: {len(metadata_df)} records")
        
        images_path = self.processed_dir / "images.parquet"
        images_df.to_parquet(images_path, index=False)
        logger.info(f"Saved images: {len(images_df)} records")
        
        # Save encoders
        user_mapping = {
            'original_to_encoded': dict(zip(self.user_encoder.classes_, range(len(self.user_encoder.classes_)))),
            'encoded_to_original': dict(zip(range(len(self.user_encoder.classes_)), self.user_encoder.classes_))
        }
        
        item_mapping = {
            'original_to_encoded': dict(zip(self.item_encoder.classes_, range(len(self.item_encoder.classes_)))),
            'encoded_to_original': dict(zip(range(len(self.item_encoder.classes_)), self.item_encoder.classes_))
        }
        
        mappings = {
            'user_mapping': user_mapping,
            'item_mapping': item_mapping,
            'n_users': len(self.user_encoder.classes_),
            'n_items': len(self.item_encoder.classes_)
        }
        
        mappings_path = self.processed_dir / "id_mappings.json"
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f, indent=2)
        logger.info("Saved ID mappings")
    
    def generate_processing_stats(self, interactions_df: pd.DataFrame, splits: Dict[str, pd.DataFrame]) -> None:
        """
        Generate and save processing statistics.
        
        Args:
            interactions_df: Full filtered interactions
            splits: Train/val/test splits
        """
        logger.info("Generating processing statistics")
        
        stats = {
            'processing_timestamp': datetime.now().isoformat(),
            'processing_parameters': {
                'min_interactions': self.min_interactions,
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio
            },
            'dataset_statistics': {
                'total_interactions': len(interactions_df),
                'unique_users': interactions_df['user_id'].nunique(),
                'unique_items': interactions_df['parent_asin'].nunique(),
                'density': len(interactions_df) / (interactions_df['user_id'].nunique() * interactions_df['parent_asin'].nunique()),
                'avg_rating': float(interactions_df['rating'].mean()),
                'rating_distribution': interactions_df['rating'].value_counts().to_dict()
            },
            'split_statistics': {}
        }
        
        for split_name, df in splits.items():
            stats['split_statistics'][split_name] = {
                'n_interactions': len(df),
                'n_users': df['user_id'].nunique(),
                'n_items': df['parent_asin'].nunique(),
                'avg_rating': float(df['rating'].mean()),
                'time_range': {
                    'start': df['datetime'].min().isoformat(),
                    'end': df['datetime'].max().isoformat()
                }
            }
        
        # Save stats
        stats_path = self.processed_dir / "processing_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Processing stats saved to {stats_path}")
        
        # Log key statistics
        logger.info("=== PROCESSING SUMMARY ===")
        logger.info(f"Total interactions: {stats['dataset_statistics']['total_interactions']:,}")
        logger.info(f"Unique users: {stats['dataset_statistics']['unique_users']:,}")
        logger.info(f"Unique items: {stats['dataset_statistics']['unique_items']:,}")
        logger.info(f"Dataset density: {stats['dataset_statistics']['density']:.6f}")
        logger.info(f"Average rating: {stats['dataset_statistics']['avg_rating']:.2f}")
        
        for split_name, split_stats in stats['split_statistics'].items():
            logger.info(f"{split_name.upper()}: {split_stats['n_interactions']:,} interactions")
    
    def run_processing(self) -> None:
        """
        Run the complete data processing pipeline.
        """
        logger.info("Starting data processing pipeline")
        
        try:
            # 1. Load data
            self.load_data()
            
            # 2. Clean and filter
            interactions_df = self.clean_and_filter_data()
            
            # 3. Create encodings
            interactions_df = self.create_user_item_encodings(interactions_df)
            
            # 4. Create chronological splits
            splits = self.create_chronological_splits(interactions_df)
            
            # 5. Build interaction matrices
            matrices = self.build_interaction_matrices(splits)
            
            # 6. Prepare metadata and images
            metadata_df = self.prepare_metadata(interactions_df)
            images_df = self.prepare_images(interactions_df)
            
            # 7. Save processed data
            self.save_processed_data(interactions_df, splits, matrices, metadata_df, images_df)
            
            # 8. Generate statistics
            self.generate_processing_stats(interactions_df, splits)
            
            logger.info("Data processing completed successfully!")
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise


def main():
    """
    Main function to run the data processing pipeline.
    """
    # Create processor with minimum 5 interactions (as per README)
    processor = AmazonBeautyProcessor(min_interactions=5)
    processor.run_processing()


if __name__ == "__main__":
    main()
