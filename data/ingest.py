"""
Data Ingestion Script for Amazon Beauty Dataset

This script downloads Amazon Beauty metadata, reviews, and image URLs
from the Hugging Face McAuley-Lab repository and saves them as Parquet files
in the interim directory for further processing.
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import json
import requests
from typing import Dict, List, Optional
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonBeautyIngester:
    """
    Handles ingestion of Amazon Beauty dataset from Hugging Face.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configuration
        self.dataset_name = "McAuley-Lab/Amazon-Reviews-2023"
        self.review_subset = "raw_review_All_Beauty"
        self.metadata_subset = "raw_meta_All_Beauty"
        
    def download_reviews(self) -> pd.DataFrame:
        """
        Download Amazon Beauty reviews dataset.
        
        Returns:
            pd.DataFrame: Reviews dataframe with columns like user_id, parent_asin, 
                         rating, timestamp, text, etc.
        """
        logger.info(f"Downloading reviews from {self.dataset_name}/{self.review_subset}")
        
        try:
            # Load reviews dataset
            reviews_dataset = load_dataset(
                self.dataset_name, 
                self.review_subset, 
                trust_remote_code=True
            )
            
            # Convert to pandas DataFrame
            reviews_df = reviews_dataset['full'].to_pandas()
            
            logger.info(f"Downloaded {len(reviews_df)} reviews")
            logger.info(f"Reviews columns: {list(reviews_df.columns)}")
            
            # Basic data info
            logger.info(f"Date range: {reviews_df['timestamp'].min()} to {reviews_df['timestamp'].max()}")
            logger.info(f"Unique users: {reviews_df['user_id'].nunique()}")
            logger.info(f"Unique items: {reviews_df['parent_asin'].nunique()}")
            
            return reviews_df
            
        except Exception as e:
            logger.error(f"Error downloading reviews: {e}")
            raise
    
    def download_metadata(self) -> pd.DataFrame:
        """
        Download Amazon Beauty metadata dataset.
        
        Returns:
            pd.DataFrame: Metadata dataframe with product information, images, etc.
        """
        logger.info(f"Downloading metadata from {self.dataset_name}/{self.metadata_subset}")
        
        try:
            # Load metadata dataset
            metadata_dataset = load_dataset(
                self.dataset_name, 
                self.metadata_subset, 
                split="full",
                trust_remote_code=True
            )
            
            # Convert to pandas DataFrame
            metadata_df = metadata_dataset.to_pandas()
            
            logger.info(f"Downloaded {len(metadata_df)} metadata records")
            logger.info(f"Metadata columns: {list(metadata_df.columns)}")
            
            # Basic data info
            logger.info(f"Unique ASINs: {metadata_df['parent_asin'].nunique()}")
            
            return metadata_df
            
        except Exception as e:
            logger.error(f"Error downloading metadata: {e}")
            raise
    
    def extract_image_urls(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and process image URLs from metadata.
        
        Args:
            metadata_df: Metadata dataframe
            
        Returns:
            pd.DataFrame: Dataframe with image URLs and related information
        """
        logger.info("Extracting image URLs from metadata")
        
        image_data = []
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing images"):
            parent_asin = row['parent_asin']
            
            # Extract images - the 'images' column contains a dictionary with resolution keys
            if 'images' in row and pd.notna(row['images']):
                images = row['images']
                
                # Handle dictionary format: {'hi_res': [url1, url2], 'large': [url3, url4]}
                if isinstance(images, dict):
                    # Priority order for image resolutions
                    resolution_priority = ['hi_res', 'large', 'medium', 'small', 'thumb']
                    
                    for resolution in resolution_priority:
                        if resolution in images and images[resolution] is not None:
                            url_list = images[resolution]
                            # Handle numpy arrays or lists
                            if hasattr(url_list, '__len__') and len(url_list) == 0:
                                continue
                            
                            # Handle numpy arrays (convert to list-like iteration)
                            if hasattr(url_list, '__iter__'):
                                for img_idx, img_url in enumerate(url_list):
                                    # Skip None values
                                    if img_url is not None and img_url != '' and str(img_url) != 'None':
                                        image_data.append({
                                            'parent_asin': parent_asin,
                                            'image_index': img_idx,
                                            'image_url': str(img_url),
                                            'image_type': resolution,
                                            'resolution': resolution
                                        })
                            elif isinstance(url_list, str) and url_list != '':
                                image_data.append({
                                    'parent_asin': parent_asin,
                                    'image_index': 0,
                                    'image_url': url_list,
                                    'image_type': resolution,
                                    'resolution': resolution
                                })
                
                # Handle legacy list format (if any)
                elif isinstance(images, list):
                    for img_idx, img_info in enumerate(images):
                        if isinstance(img_info, dict):
                            image_data.append({
                                'parent_asin': parent_asin,
                                'image_index': img_idx,
                                'image_url': img_info.get('large', img_info.get('medium', img_info.get('small', ''))),
                                'image_type': 'product',
                                'resolution': 'unknown'
                            })
                        elif isinstance(img_info, str) and img_info != '':
                            image_data.append({
                                'parent_asin': parent_asin,
                                'image_index': img_idx,
                                'image_url': img_info,
                                'image_type': 'product',
                                'resolution': 'unknown'
                            })
                
                # Handle single string format
                elif isinstance(images, str) and images != '':
                    image_data.append({
                        'parent_asin': parent_asin,
                        'image_index': 0,
                        'image_url': images,
                        'image_type': 'product',
                        'resolution': 'unknown'
                    })
        
        image_df = pd.DataFrame(image_data)
        
        if len(image_df) > 0:
            # Remove empty URLs
            image_df = image_df[image_df['image_url'].str.len() > 0]
            
            logger.info(f"Extracted {len(image_df)} image URLs")
            logger.info(f"Images for {image_df['parent_asin'].nunique()} unique products")
            
            # Log distribution by resolution
            if 'resolution' in image_df.columns:
                resolution_counts = image_df['resolution'].value_counts()
                logger.info(f"Resolution distribution: {resolution_counts.to_dict()}")
        else:
            logger.warning("No image URLs found in metadata")
        
        return image_df
    
    def save_as_parquet(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save dataframe as Parquet file in interim directory.
        
        Args:
            df: Dataframe to save
            filename: Output filename (without extension)
        """
        output_path = self.interim_dir / f"{filename}.parquet"
        
        logger.info(f"Saving {len(df)} records to {output_path}")
        
        try:
            df.to_parquet(output_path, index=False)
            logger.info(f"Successfully saved {filename}.parquet")
            
            # Log file size
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"File size: {file_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"Error saving {filename}.parquet: {e}")
            raise
    
    def run_ingestion(self) -> None:
        """
        Run the complete data ingestion pipeline.
        """
        logger.info("Starting Amazon Beauty data ingestion")
        
        try:
            # 1. Download reviews
            reviews_df = self.download_reviews()
            self.save_as_parquet(reviews_df, "reviews")
            
            # 2. Download metadata
            metadata_df = self.download_metadata()
            self.save_as_parquet(metadata_df, "metadata")
            
            # 3. Extract image URLs
            image_df = self.extract_image_urls(metadata_df)
            if len(image_df) > 0:
                self.save_as_parquet(image_df, "images")
            
            # 4. Generate summary statistics
            self.generate_summary_stats(reviews_df, metadata_df, image_df)
            
            logger.info("Data ingestion completed successfully!")
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise
    
    def generate_summary_stats(self, reviews_df: pd.DataFrame, metadata_df: pd.DataFrame, 
                             image_df: pd.DataFrame) -> None:
        """
        Generate and save summary statistics.
        
        Args:
            reviews_df: Reviews dataframe
            metadata_df: Metadata dataframe
            image_df: Images dataframe
        """
        logger.info("Generating summary statistics")
        
        stats = {
            'ingestion_timestamp': pd.Timestamp.now().isoformat(),
            'reviews': {
                'total_reviews': len(reviews_df),
                'unique_users': reviews_df['user_id'].nunique(),
                'unique_items': reviews_df['parent_asin'].nunique(),
                'date_range': {
                    'min': reviews_df['timestamp'].min(),
                    'max': reviews_df['timestamp'].max()
                },
                'rating_distribution': reviews_df['rating'].value_counts().to_dict(),
                'avg_rating': float(reviews_df['rating'].mean())
            },
            'metadata': {
                'total_products': len(metadata_df),
                'unique_asins': metadata_df['parent_asin'].nunique(),
                'products_with_images': len(image_df['parent_asin'].unique()) if len(image_df) > 0 else 0
            },
            'images': {
                'total_image_urls': len(image_df),
                'unique_products_with_images': image_df['parent_asin'].nunique() if len(image_df) > 0 else 0
            }
        }
        
        # Save stats as JSON
        stats_path = self.interim_dir / "ingestion_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Summary stats saved to {stats_path}")
        
        # Log key statistics
        logger.info(f"Total reviews: {stats['reviews']['total_reviews']:,}")
        logger.info(f"Unique users: {stats['reviews']['unique_users']:,}")
        logger.info(f"Unique items: {stats['reviews']['unique_items']:,}")
        logger.info(f"Total products: {stats['metadata']['total_products']:,}")
        logger.info(f"Total image URLs: {stats['images']['total_image_urls']:,}")


def main():
    """
    Main function to run the data ingestion pipeline.
    """
    ingester = AmazonBeautyIngester()
    ingester.run_ingestion()


if __name__ == "__main__":
    main() 