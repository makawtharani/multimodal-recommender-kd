# Data Ingestion for Amazon Beauty Dataset

This directory contains the data ingestion pipeline for the Amazon Beauty dataset from Hugging Face.

## Files

- `ingest.py` - Main data ingestion script
- `raw/` - Raw data files (populated after ingestion)
- `interim/` - Processed Parquet files (populated after ingestion)
- `processed/` - Final processed data for training (populated in later steps)

## Usage

### Run the full ingestion pipeline:

```bash
# Activate virtual environment
source venv/bin/activate

# Run ingestion
python data/ingest.py
```

### Import as a module:

```python
from data.ingest import AmazonBeautyIngester

# Initialize ingester
ingester = AmazonBeautyIngester()

# Run full pipeline
ingester.run_ingestion()

# Or run individual steps
reviews_df = ingester.download_reviews()
metadata_df = ingester.download_metadata()
image_df = ingester.extract_image_urls(metadata_df)
```

## Output Files

After running the ingestion, the following files will be created in `data/interim/`:

- `reviews.parquet` - User reviews with ratings, text, timestamps
- `metadata.parquet` - Product metadata with descriptions, categories, etc.
- `images.parquet` - Image URLs extracted from product metadata
- `ingestion_stats.json` - Summary statistics about the ingested data

## Data Schema

### Reviews (`reviews.parquet`)
- `user_id` - Unique user identifier
- `parent_asin` - Product identifier
- `rating` - User rating (1-5)
- `timestamp` - Review timestamp
- `text` - Review text content
- Additional columns from the original dataset

### Metadata (`metadata.parquet`)
- `parent_asin` - Product identifier
- `title` - Product title
- `description` - Product description
- `categories` - Product categories
- `brand` - Product brand
- Additional columns from the original dataset

### Images (`images.parquet`)
- `parent_asin` - Product identifier
- `image_index` - Image index (-1 for main image)
- `image_url` - Direct URL to product image
- `image_type` - Type of image ('main' or 'product')

## Dataset Information

The script downloads data from:
- **Dataset**: McAuley-Lab/Amazon-Reviews-2023
- **Reviews subset**: raw_review_All_Beauty
- **Metadata subset**: raw_meta_All_Beauty

This is a filtered subset of Amazon product reviews and metadata specifically for beauty products.

## Requirements

The ingestion script requires:
- `datasets` - For Hugging Face dataset loading
- `pandas` - For data manipulation
- `pyarrow` - For Parquet file format
- `tqdm` - For progress bars
- `requests` - For HTTP requests (if needed)

All dependencies are listed in the main `requirements.txt` file. 